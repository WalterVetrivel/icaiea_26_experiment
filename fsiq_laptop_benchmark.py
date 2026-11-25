"""
fsiq_laptop_benchmark.py - speed and memory test for local models on YOUR laptop
=================================================================================
Companion to the Colab notebook "FailureSensorIQ_local_LLM_experiments.ipynb".

The notebook measures ACCURACY on a fast cloud GPU. This script measures what
a small firm would actually experience on a cheap, CPU-only computer:
seconds per question, tokens per second, cold-start time and RAM use. It uses
the SAME questions, prompts, retrieval and answer parsing as the notebook
(those parts are copied in unchanged), so the two sets of results line up.

ONE-TIME SETUP (Windows, in a Command Prompt or PowerShell):
    pip install requests pandas openpyxl rank_bm25 psutil tqdm
    ollama pull qwen3.5:2b
    ollama pull qwen3.5:4b

BEFORE RUNNING:
    * Copy sample_ids.csv from your Google Drive folder (the notebook creates
      it) into the same folder as this script.
    * Close other heavy programs (browser tabs!) - with 8 GB RAM they compete
      with the model for memory and will distort the timings.
    * Plug the laptop in; on battery, Windows slows the CPU down.

RUN (about 100 questions x 2 models x 2 conditions; expect 30-90 minutes):
    python fsiq_laptop_benchmark.py --models qwen3.5:2b qwen3.5:4b

    Optional - check the laptop gives the same answers as Colab (download the
    Colab "results" folder from Drive first):
    python fsiq_laptop_benchmark.py --models qwen3.5:4b --colab-results results

OUTPUT (folder laptop_results):
    one CSV per model/condition with every answer and timing, plus
    laptop_summary.csv with one row per model/condition - this is the table
    for the "deployment on low-cost hardware" part of the paper.

Like the notebook, it is resume-safe: if you stop it, run the same command
again and it continues where it left off. (Memory figures are measured only
for questions answered in the current run.)
"""
import os
import platform
import threading
import time


# =====================================================================
# PART A - LOADING THE FAILURESENSORIQ QUESTIONS
# =====================================================================
# FailureSensorIQ (IBM Research, NeurIPS 2025) has 2,667 single-answer
# multiple-choice questions about which sensors detect which failure
# modes, across 10 industrial assets. Licence: CC BY 4.0 - cite the paper.
#
# The same 2,667 questions come in three versions. All three share the
# same question "id", so we can line them up and check whether a model
# answers the SAME question correctly in every version:
#   original     - the questions as written
#   simple_pert  - options shuffled and labels changed from A,B,C to P,Q,R
#   complex_pert - as above, plus the question reworded by another LLM
# A model that really "knows" the answer should get all three right.
# A model that is guessing or relying on option position will not.
# =====================================================================
import json
import os
import random

import pandas as pd

VARIANT_FILES = {
    "original":     "eval_data/fmsr_processed/fmsr_filtered_data_all.jsonl",
    "simple_pert":  "eval_data/fmsr_processed/fmsr_filtered_perturbed_data_all_simple.jsonl",
    "complex_pert": "eval_data/fmsr_processed/fmsr_filtered_perturbed_data_all_llama.jsonl",
}


def read_jsonl(path):
    """Read a .jsonl file (one JSON object per line) into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(repo_dir, variant="original"):
    """Load one version of the questions as a table (pandas DataFrame).

    Each row is one question with these columns:
      id        - question id (identical across the three versions)
      asset     - e.g. "electric motor" (perturbed files lack this, so we
                  copy it over from the original file using the id)
      question  - the question text
      labels    - option letters, e.g. ["A","B","C","D","E"] or ["P","Q",...]
      options   - option texts, in the same order as labels
      answer    - the letter of the correct option
      n_options - number of options (2 to 5); random guessing = 1/n_options
    """
    original = read_jsonl(os.path.join(repo_dir, VARIANT_FILES["original"]))
    asset_by_id = {r["id"]: r["asset_name"] for r in original}

    if variant == "original":
        raw = original
    else:
        raw = read_jsonl(os.path.join(repo_dir, VARIANT_FILES[variant]))

    records = []
    for r in raw:
        # Perturbed files write labels as "P)", "Q)" ... so strip the bracket.
        labels = [lab.strip().rstrip(")").strip() for lab in r["option_ids"]]
        correct_positions = [i for i, flag in enumerate(r["correct"]) if flag]
        assert len(correct_positions) == 1, "expected exactly one correct option"
        records.append({
            "id": int(r["id"]),
            "asset": asset_by_id[r["id"]],
            "question": r["question"].strip(),
            "labels": labels,
            "options": [str(o).strip() for o in r["options"]],
            "answer": labels[correct_positions[0]],
            "n_options": len(labels),
        })
    return pd.DataFrame(records)


def stratified_sample(df, n, seed=42):
    """Pick n questions so each asset keeps its share of the full dataset.

    Example: power transformers are 544 of 2,667 questions (20.4%), so a
    1,000-question sample gets about 204 of them. The fixed seed means you
    get the SAME sample every time, on Colab and on your laptop.
    """
    if n >= len(df):
        return df.copy()
    shares = df["asset"].value_counts() / len(df)
    exact = shares * n
    counts = exact.astype(int)
    # Hand out the remaining places to the assets with the largest
    # fractional parts, so the counts add up to exactly n.
    remainder = int(n - counts.sum())
    for asset in (exact - counts).sort_values(ascending=False).index[:remainder]:
        counts[asset] += 1
    rng = random.Random(seed)
    picked = []
    for asset, k in counts.items():
        ids = sorted(df.loc[df["asset"] == asset, "id"].tolist())
        picked.extend(rng.sample(ids, int(k)))
    return df[df["id"].isin(picked)].sort_values("id").reset_index(drop=True)


# =====================================================================
# PART B - THE KNOWLEDGE BASE AND RETRIEVER (the "R" in RAG)
# =====================================================================
# RAG = Retrieval-Augmented Generation. Before asking the model a
# question, we search a document collection for the most relevant
# passages and paste them into the prompt. The model then answers using
# those notes, without needing the knowledge stored in its own weights.
#
# Our document collection is the ISO 17359 condition-monitoring table
# that IBM released with FailureSensorIQ (file iso_sensors.xlsx). It has
# one sheet per asset: rows are faults, columns are sensors, and an "X"
# means "this sensor can detect this fault". In the paper we frame this
# as "the firm's own condition-monitoring documentation". Many plants
# keep exactly this kind of FMEA/CM table.
#
# IMPORTANT for the paper: the questions were generated from this same
# table, so RAG with 100% of the table is a best case ("complete
# documentation"). That is why we also run a "coverage" setting that
# randomly drops some fault rows (e.g. coverage=0.5 keeps half). It
# shows how accuracy falls when the firm's documentation is incomplete,
# which is the realistic case.
#
# We turn the table into short text passages ("chunks") of two kinds:
#   fault chunk : asset + fault -> which sensors detect it / which don't
#   sensor chunk: asset + sensor -> which faults it indicates / which not
# Both are needed because questions run in both directions.
#
# Retrieval uses BM25, a classic keyword-matching search algorithm (the
# same family search engines used before neural search). It needs no
# GPU, no extra model and runs instantly on a CPU laptop, which suits a
# low-cost deployment.
#
# We search only each passage's TITLE (e.g. "pump | fault: bearing
# wear"), using the question text as the query. Searching the full
# passages works worse, because every passage also lists many sensors
# and faults in its "NOT associated" part, which confuses keyword
# matching. Tested on the 1,000-question sample, title search puts the
# right passage in the top 4 for 99% of original questions and 97.5% of
# reworded ones (vs about 95% for full-text search). Worth one sentence
# in the Methods section.
# =====================================================================
import random
import re

import pandas as pd
from rank_bm25 import BM25Okapi

KB_FILE = "LLM-Embeddings/data/raw/iso_sensors.xlsx"
OVERVIEW_SHEET = "Condition Monitoring Parameter"


def _clean(text):
    """Lower-case and tidy a table label, e.g. 'Rotor windings fault '."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _sheet_to_asset(sheet_name, asset_names):
    """Match an Excel sheet name to the asset name used in the questions.

    Excel cuts sheet names at 31 characters, so the sheet for
    "reciprocating internal combustion engine" is called
    "reciprocating internal combusti". We match by prefix.
    """
    s = _clean(sheet_name)
    for asset in asset_names:
        if asset == s or asset.startswith(s):
            return asset
    return None


def build_knowledge_chunks(xlsx_path, asset_names, coverage=1.0, seed=0):
    """Turn the ISO table into a list of text passages for retrieval.

    coverage: fraction of fault rows to keep (1.0 = full documentation,
              0.5 = half the faults are undocumented). Rows are dropped
              at random but reproducibly (fixed seed).
    """
    xl = pd.ExcelFile(xlsx_path)
    rng = random.Random(seed)
    chunks = []

    # 1) Overview sheet: which parameters are monitored for each asset.
    overview = xl.parse(OVERVIEW_SHEET)
    for col in overview.columns[1:]:
        asset = _sheet_to_asset(col, asset_names)
        if asset is None:
            continue
        params = [_clean(p) for p, mark in zip(overview.iloc[:, 0], overview[col])
                  if isinstance(mark, str) and mark.strip()]
        chunks.append(f"{asset} | condition-monitoring parameters used for this asset: "
                      + ", ".join(params) + ".")

    # 2) One sheet per asset: fault rows x sensor columns.
    for sheet in xl.sheet_names:
        if sheet == OVERVIEW_SHEET:
            continue
        asset = _sheet_to_asset(sheet, asset_names)
        if asset is None:
            continue
        table = xl.parse(sheet)
        sensors = [_clean(c) for c in table.columns[1:]]

        # Read each fault row: which sensors have an "X".
        rows = []
        for _, r in table.iterrows():
            fault = _clean(r.iloc[0])
            marks = [isinstance(v, str) and v.strip() != "" for v in r.iloc[1:]]
            rows.append((fault, marks))

        # Simulate incomplete documentation by dropping some fault rows.
        if coverage < 1.0:
            keep = max(1, round(len(rows) * coverage))
            rows = rng.sample(rows, keep)

        # Fault chunks: fault -> sensors.
        for fault, marks in rows:
            yes = [s for s, m in zip(sensors, marks) if m]
            no = [s for s, m in zip(sensors, marks) if not m]
            chunks.append(
                f"{asset} | fault: {fault}. Sensors that can detect this fault: "
                f"{', '.join(yes) or 'none listed'}. "
                f"Sensors NOT associated with this fault: {', '.join(no) or 'none'}.")

        # Sensor chunks: sensor -> faults (built only from the kept rows).
        for j, sensor in enumerate(sensors):
            yes = [f for f, marks in rows if marks[j]]
            no = [f for f, marks in rows if not marks[j]]
            chunks.append(
                f"{asset} | sensor: {sensor}. Faults indicated by abnormal {sensor} "
                f"readings: {', '.join(yes) or 'none listed'}. "
                f"Faults NOT indicated by this sensor: {', '.join(no) or 'none'}.")
    return chunks


def _tokenize(text):
    """Split text into lower-case words for BM25 keyword matching."""
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_title(chunk):
    """The title is everything before the first '. ', e.g.
    'electric motor | fault: bearing damage'."""
    return chunk.split(". ", 1)[0]


class BM25Retriever:
    """Keyword search over the knowledge chunks (matching on titles)."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(chunk_title(c)) for c in chunks])

    def retrieve(self, query, k=4):
        """Return the k chunks whose titles best match the query text."""
        scores = self.bm25.get_scores(_tokenize(query))
        best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in best]


# =====================================================================
# PART C - PROMPTS AND ANSWER PARSING
# =====================================================================
# The four experimental conditions differ ONLY in the prompt:
#   zero_shot - question + options, "reply with the letter only"
#   cot       - "chain of thought": reason briefly, then give the letter
#   rag       - retrieved notes + question, "reply with the letter only"
#   rag_cot   - retrieved notes + brief reasoning
# Keeping everything else identical (same model, same settings) means
# any difference in accuracy comes from the method.
#
# !! These prompts are copied word-for-word into the laptop script.
# !! If you change them here, change them there too, or the Colab and
# !! laptop results will not be comparable.
# =====================================================================
import re

SYSTEM_PROMPT = (
    "You are an experienced reliability and condition-monitoring engineer. "
    "You answer multiple-choice questions about industrial assets, their "
    "failure modes and the sensors used to detect them."
)

LETTER_ONLY = ("Reply with only the letter of the correct option "
               "(for example: {example}). Do not explain.")
REASON_THEN_ANSWER = ("Think through the question step by step in a few short "
                      "sentences, then write the final line exactly as "
                      "'Answer: <letter>'.")


def format_options(labels, options):
    """Lay the options out one per line: 'A. vibration'."""
    return "\n".join(f"{lab}. {opt}" for lab, opt in zip(labels, options))


def retrieval_query(row):
    """Text used to search the knowledge base: the question only.
    (Adding the option texts made retrieval worse in testing, because
    options mention sensors/faults from many different passages.)"""
    return row["question"]


def build_messages(row, condition, context_chunks=None):
    """Build the chat messages sent to the model for one question.

    row       - one question (a row of the DataFrame from Part A)
    condition - "zero_shot", "cot", "rag" or "rag_cot"
    context_chunks - retrieved passages (only for the rag conditions)
    """
    parts = []
    if condition.startswith("rag"):
        notes = "\n".join(f"- {c}" for c in (context_chunks or []))
        parts.append("Reference notes from the plant's condition-monitoring "
                     "documentation (they may be incomplete):\n" + notes)
    parts.append("Question: " + row["question"])
    parts.append("Options:\n" + format_options(row["labels"], row["options"]))
    if condition.endswith("cot"):
        parts.append(REASON_THEN_ANSWER)
    else:
        parts.append(LETTER_ONLY.format(example=row["labels"][0]))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_answer(text, labels, options):
    """Pull the chosen letter out of the model's reply.

    Small models do not always follow instructions exactly, so we try
    several patterns, from most to least specific. Returns None if no
    answer can be found. That counts as WRONG, and we also report the
    parse-failure rate, because "cannot follow the format" is itself a
    finding about small models.
    """
    if not text:
        return None
    valid = set(labels)
    letters = "|".join(re.escape(l) for l in labels)

    # 1) "Answer: B" / "answer is (B)" - take the LAST one (after reasoning).
    #    "answer" may be any case, but the letter must be a capital, so
    #    "the answer is a pressure sensor" is NOT read as option A.
    hits = re.findall(rf"(?i:answer)\s*(?i:is)?\s*[:\-]?\s*\**\(?({letters})\)?(?![A-Za-z])",
                      text)
    hits = [h for h in hits if h in valid]
    if hits:
        return hits[-1]

    # 2) The reply is just a letter ("B", "(B)", "**B**") or starts with
    #    a letter followed by punctuation ("B. vibration", "B) vibration").
    m = re.match(rf"^\**\(?({letters})\)?\**\s*$", text.strip())
    if not m:
        m = re.match(rf"^\**\(?({letters})[.):\-]", text.strip())
    if m:
        return m.group(1)

    # 3) The reply quotes exactly one option's text, e.g. "vibration".
    lowered = text.lower()
    quoted = [lab for lab, opt in zip(labels, options)
              if re.search(rf"\b{re.escape(opt.lower())}\b", lowered)]
    if len(quoted) == 1:
        return quoted[0]
    return None


# =====================================================================
# PART D - TALKING TO OLLAMA
# =====================================================================
# Ollama runs a small web server (default http://localhost:11434). We
# send it a question as JSON and it sends back the model's reply plus
# timing statistics. The same function works on Colab and your laptop.
#
# Settings that matter for a fair experiment:
#   temperature 0 - always pick the most likely word, so the model gives
#                   (almost) the same answer every run
#   seed 42       - fixes any remaining randomness
#   think False   - Qwen 3.5 and Gemma 4 have a "thinking mode" that writes
#                   long hidden reasoning before answering. We switch it
#                   off so that "reasoning" is controlled only by our
#                   cot prompt, the same way for every model. (If a model
#                   rejects the setting, we retry without it.)
#   num_ctx 4096  - context window in tokens; plenty for our prompts and
#                   keeps memory use low on an 8 GB laptop
# =====================================================================
import re
import time

import requests

OLLAMA_HOST = "http://localhost:11434"
_MODELS_WITHOUT_THINK_SETTING = set()


def ollama_chat(model, messages, max_tokens, host=None, timeout=900):
    """Send one chat request to Ollama. Returns the reply and timings.

    Returned dict:
      text          - the model's reply (thinking text removed)
      wall_s        - total seconds from sending to receiving (what a
                      user would actually wait)
      prompt_tokens - tokens in the prompt; output_tokens - tokens generated
      prompt_s      - seconds spent reading the prompt
      gen_s         - seconds spent writing the reply
      load_s        - seconds spent loading the model into memory (only
                      large on the first call after a model switch)
    """
    host = host or OLLAMA_HOST
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",        # keep the model loaded between questions
        "options": {"temperature": 0, "seed": 42,
                    "num_predict": max_tokens, "num_ctx": 4096},
    }
    if model not in _MODELS_WITHOUT_THINK_SETTING:
        payload["think"] = False

    start = time.perf_counter()
    resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
    if resp.status_code == 400 and "think" in payload and "think" in resp.text.lower():
        # This model does not accept the think setting - retry without it.
        _MODELS_WITHOUT_THINK_SETTING.add(model)
        return ollama_chat(model, messages, max_tokens, host, timeout)
    resp.raise_for_status()
    data = resp.json()
    wall = time.perf_counter() - start

    text = data.get("message", {}).get("content", "") or ""
    # Some models still put reasoning inside <think>...</think>; drop it.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    ns = 1e9  # Ollama reports durations in nanoseconds
    return {
        "text": text,
        "wall_s": wall,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "prompt_s": data.get("prompt_eval_duration", 0) / ns,
        "gen_s": data.get("eval_duration", 0) / ns,
        "load_s": data.get("load_duration", 0) / ns,
    }


# =====================================================================
# PART E - RUNNING ONE EXPERIMENTAL CONDITION (resume-safe)
# =====================================================================
# run_condition() asks one model every question in a sample under one
# condition and writes one CSV row per question.
#
# Resume-safe: every answer is written to the CSV the moment it arrives.
# If Colab disconnects, just re-run. Questions already in the CSV are
# skipped, so you lose at most the few in progress. Questions that hit
# an error are NOT written, so they are retried on the next run.
#
# Parallel requests: on a GPU, Ollama can answer several questions at
# once (set by OLLAMA_NUM_PARALLEL), which is much faster. On the
# laptop we use workers=1 to measure what a single user would see.
# =====================================================================
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm.auto import tqdm   # progress bar, if installed
except ImportError:              # pragma: no cover
    tqdm = None

# Maximum reply length (tokens). Letter-only answers need very few;
# reasoning answers need room to think.
MAX_TOKENS = {"zero_shot": 16, "rag": 16, "cot": 512, "rag_cot": 512}

RESULT_COLUMNS = [
    "id", "asset", "variant", "model", "condition", "rag_coverage",
    "answer", "pred", "correct", "parsed", "n_options",
    "wall_s", "prompt_tokens", "output_tokens", "prompt_s", "gen_s", "load_s",
    "raw",
]


def results_file(out_dir, model, condition, variant, rag_coverage):
    """One CSV per (model, condition, version, coverage). Example:
    results/qwen3.5_4b__rag_cov50__original.csv"""
    safe_model = model.replace(":", "_").replace("/", "_")
    cond = condition
    if condition.startswith("rag"):
        cond = f"{condition}_cov{int(round(rag_coverage * 100))}"
    return os.path.join(out_dir, f"{safe_model}__{cond}__{variant}.csv")


def _done_ids(path):
    """Question ids already answered in an existing results file."""
    if not os.path.exists(path):
        return set()
    try:
        return set(pd.read_csv(path, usecols=["id"])["id"].astype(int))
    except Exception:
        return set()


def run_condition(sample, model, condition, variant, out_dir,
                  retriever=None, rag_coverage=1.0, rag_k=4,
                  workers=4, host=None, show_progress=True):
    """Answer every question in `sample` and save results to CSV.

    sample    - DataFrame of questions (from Part A)
    model     - Ollama model tag, e.g. "qwen3.5:4b"
    condition - "zero_shot", "cot", "rag" or "rag_cot"
    variant   - "original", "simple_pert" or "complex_pert" (just a label)
    retriever - a BM25Retriever (needed for the rag conditions)
    rag_coverage - which documentation level the retriever was built with
                   (only used for naming the file)
    Returns the full results table for this condition.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = results_file(out_dir, model, condition, variant, rag_coverage)
    done = _done_ids(path)
    todo = sample[~sample["id"].isin(done)]
    if todo.empty:
        print(f"  already complete: {os.path.basename(path)}")
        return pd.read_csv(path)
    if condition.startswith("rag") and retriever is None:
        raise ValueError("rag conditions need a retriever")

    def answer_one(row):
        context = retriever.retrieve(retrieval_query(row), k=rag_k) \
            if condition.startswith("rag") else None
        messages = build_messages(row, condition, context)
        res = ollama_chat(model, messages, MAX_TOKENS[condition], host=host)
        pred = parse_answer(res["text"], row["labels"], row["options"])
        return {
            "id": row["id"], "asset": row["asset"], "variant": variant,
            "model": model, "condition": condition,
            "rag_coverage": rag_coverage if condition.startswith("rag") else "",
            "answer": row["answer"], "pred": pred if pred else "",
            "correct": int(pred == row["answer"]), "parsed": int(pred is not None),
            "n_options": row["n_options"],
            "wall_s": round(res["wall_s"], 3),
            "prompt_tokens": res["prompt_tokens"], "output_tokens": res["output_tokens"],
            "prompt_s": round(res["prompt_s"], 3), "gen_s": round(res["gen_s"], 3),
            "load_s": round(res["load_s"], 3),
            "raw": res["text"][:2000],
        }

    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    lock = threading.Lock()
    errors = 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new_file:
            writer.writeheader()
        rows = [r for _, r in todo.iterrows()]
        bar = tqdm(total=len(rows), desc=os.path.basename(path)) \
            if (show_progress and tqdm) else None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(answer_one, r) for r in rows]
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    with lock:
                        writer.writerow(result)
                        f.flush()   # write to disk now, so a disconnect loses nothing
                except Exception as e:   # network hiccup, timeout, etc.
                    errors += 1
                    if errors <= 3:
                        print(f"  error (will retry on next run): {e}")
                if bar:
                    bar.update(1)
        if bar:
            bar.close()
    if errors:
        print(f"  {errors} questions failed - run again to retry them (finished answers are kept).")
    return pd.read_csv(path)


# =====================================================================
# PART G - LAPTOP MEASUREMENT (what an SME workstation experiences)
# =====================================================================
import argparse
import statistics
import sys

import psutil

RAW_BASE = "https://raw.githubusercontent.com/IBM/FailureSensorIQ/main/"
NEEDED_FILES = [VARIANT_FILES["original"], KB_FILE]


def ensure_data(data_dir):
    """Download the two files we need from IBM's GitHub (first run only)."""
    for rel in NEEDED_FILES:
        dest = os.path.join(data_dir, rel)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"Downloading {rel} ...")
        r = requests.get(RAW_BASE + rel, timeout=120)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)


class MemoryMonitor:
    """Checks RAM use every 0.5 s in the background while the model runs.

    Records:
      peak_ollama_gb    - highest RAM used by all Ollama processes together
                          (the model itself lives here)
      min_available_gb  - lowest free RAM left for everything else on the
                          laptop; near zero means the laptop was swapping
                          to disk, which makes everything slow
    """

    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_ollama = 0
        self.min_available = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _ollama_rss(self):
        total = 0
        for p in psutil.process_iter(["name", "memory_info"]):
            try:
                name = (p.info["name"] or "").lower()
                if name.startswith("ollama") and p.info["memory_info"]:
                    total += p.info["memory_info"].rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak_ollama = max(self.peak_ollama, self._ollama_rss())
            self.min_available = min(self.min_available, psutil.virtual_memory().available)
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def peak_ollama_gb(self):
        return self.peak_ollama / 1e9

    @property
    def min_available_gb(self):
        return self.min_available / 1e9


def percentile(values, q):
    """Simple percentile without extra libraries (q between 0 and 100)."""
    values = sorted(values)
    if not values:
        return float("nan")
    idx = min(len(values) - 1, max(0, round(q / 100 * (len(values) - 1))))
    return values[idx]


def main():
    ap = argparse.ArgumentParser(
        description="Measure speed and memory of local models on this computer.")
    ap.add_argument("--models", nargs="+", default=["qwen3.5:2b", "qwen3.5:4b"],
                    help="Ollama model tags, exactly as shown by 'ollama list'")
    ap.add_argument("--conditions", nargs="+", default=["zero_shot", "rag"],
                    choices=["zero_shot", "cot", "rag", "rag_cot"])
    ap.add_argument("--n", type=int, default=100,
                    help="questions per model and condition (100 is plenty for timing)")
    ap.add_argument("--sample-file", default="sample_ids.csv",
                    help="sample_ids.csv saved by the Colab notebook (same questions)")
    ap.add_argument("--data-dir", default="fsiq_data")
    ap.add_argument("--out", default="laptop_results")
    ap.add_argument("--colab-results", default=None,
                    help="optional: folder with the Colab results CSVs, to check "
                         "that laptop and Colab give the same answers")
    ap.add_argument("--host", default=OLLAMA_HOST)
    args = ap.parse_args()

    ensure_data(args.data_dir)
    questions = load_questions(args.data_dir, "original")

    # Use the same questions as Colab: take the Colab sample, then a smaller
    # stratified subset of it (same seed, so it is reproducible).
    if os.path.exists(args.sample_file):
        ids = set(pd.read_csv(args.sample_file)["id"].astype(int))
        questions = questions[questions["id"].isin(ids)]
        print(f"Using {len(questions)} questions from {args.sample_file}")
    else:
        print(f"WARNING: {args.sample_file} not found - sampling from all questions.")
    sample = stratified_sample(questions, args.n, seed=7)

    asset_names = sorted(questions["asset"].unique())
    retriever = BM25Retriever(build_knowledge_chunks(
        os.path.join(args.data_dir, KB_FILE), asset_names, coverage=1.0))

    machine = {
        "cpu": platform.processor() or platform.machine(),
        "cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": f"{platform.system()} {platform.release()}",
    }
    print("Machine:", machine)

    os.makedirs(args.out, exist_ok=True)
    summary_rows = []
    for model in args.models:
        # Warm-up: the first request loads the model from disk into RAM.
        # We time it separately ("cold start") so it doesn't distort the
        # per-question latency.
        print(f"\n=== {model}: loading model (cold start) ...")
        warm = ollama_chat(model, [{"role": "user", "content": "Reply with OK."}],
                           max_tokens=4, host=args.host)
        print(f"    cold start {warm['wall_s']:.1f} s (of which loading {warm['load_s']:.1f} s)")

        for condition in args.conditions:
            print(f"--- {model} / {condition}: {len(sample)} questions, one at a time")
            with MemoryMonitor() as mem:
                res = run_condition(sample, model, condition, "original", args.out,
                                    retriever=retriever, rag_coverage=1.0,
                                    workers=1, host=args.host)
            res = res[res["id"].isin(sample["id"])]
            gen_tps = [o / g for o, g in zip(res["output_tokens"], res["gen_s"]) if g > 0]
            prompt_tps = [p / s for p, s in zip(res["prompt_tokens"], res["prompt_s"]) if s > 0]
            row = {
                "model": model, "condition": condition, "n": len(res),
                "accuracy": res["correct"].mean(),
                "parse_rate": res["parsed"].mean(),
                "median_latency_s": statistics.median(res["wall_s"]),
                "p95_latency_s": percentile(list(res["wall_s"]), 95),
                "gen_tokens_per_s": statistics.median(gen_tps) if gen_tps else float("nan"),
                "prompt_tokens_per_s": statistics.median(prompt_tps) if prompt_tps else float("nan"),
                "cold_start_s": round(warm["wall_s"], 2),
                # Memory is only measured for questions run in THIS session;
                # if the CSV was already complete, these stay at 0 / inf.
                "peak_ollama_ram_gb": round(mem.peak_ollama_gb, 2),
                "min_free_ram_gb": round(mem.min_available_gb, 2),
                **machine,
            }
            summary_rows.append(row)
            print(f"    accuracy {row['accuracy']:.1%} | median {row['median_latency_s']:.1f} s/question"
                  f" | {row['gen_tokens_per_s']:.1f} tokens/s | peak Ollama RAM {row['peak_ollama_ram_gb']} GB")

    summary = pd.DataFrame(summary_rows)
    out_csv = os.path.join(args.out, "laptop_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(f"\nSaved summary to {out_csv}")

    # Optional check: does the laptop give the same answers as Colab?
    # Same model file + temperature 0 should agree on nearly every question.
    # Report the agreement rate in the paper.
    if args.colab_results:
        print("\nAgreement with Colab answers (same questions):")
        for model in args.models:
            for condition in args.conditions:
                lap = pd.read_csv(results_file(args.out, model, condition, "original", 1.0))
                col_path = results_file(args.colab_results, model, condition, "original", 1.0)
                if not os.path.exists(col_path):
                    print(f"  {model} / {condition}: no Colab file found")
                    continue
                col = pd.read_csv(col_path)
                merged = lap.merge(col, on="id", suffixes=("_laptop", "_colab"))
                same = (merged["pred_laptop"].fillna("") == merged["pred_colab"].fillna("")).mean()
                print(f"  {model} / {condition}: {same:.1%} identical answers on {len(merged)} questions")


if __name__ == "__main__":
    sys.exit(main())
