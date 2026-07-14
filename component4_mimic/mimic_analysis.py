"""
Component 4: MIMIC-III Radiology Notes Analysis
================================================
Applies the same linguistic feature extraction (Component 2) and tokenizer
fragmentation analysis (Component 3) to MIMIC-III radiology notes, then
compares results against the ClinicalTrials.gov baseline.

Input:  NOTEEVENTS.csv (path configurable via --noteevents, default data/raw/NOTEEVENTS.csv)
        data/processed/linguistic_features.csv      (Component 2 baseline)
        data/processed/tokenizer_fragmentation.csv  (Component 3 baseline)

Output: data/processed/mimic_linguistic_features.csv
        data/processed/mimic_tokenizer_fragmentation.csv
"""

import re
import sys
import argparse
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

# ── Device ────────────────────────────────────────────────────────────────────
# MPS = Apple Silicon GPU. Tokenizer.tokenize() is CPU-only; this device flag
# is wired in for any future model inference added to this pipeline.
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"

TRIALS_LING_PATH  = DATA_DIR / "processed" / "linguistic_features.csv"
TRIALS_FRAG_PATH  = DATA_DIR / "processed" / "tokenizer_fragmentation.csv"
MIMIC_LING_OUT    = DATA_DIR / "processed" / "mimic_linguistic_features.csv"
MIMIC_FRAG_OUT    = DATA_DIR / "processed" / "mimic_tokenizer_fragmentation.csv"

SAMPLE_N      = 500
RANDOM_STATE  = 42
CHUNK_SIZE    = 10_000

# ── Word lists (identical to Component 2) ────────────────────────────────────
HEDGING_WORDS = {
    "possible", "possibly", "may", "might", "could", "suggest", "appears",
    "seems", "potential", "potentially", "unclear", "uncertain", "likely",
    "unlikely", "whether", "approximately", "evaluate", "assess",
}

NEGATION_WORDS = {
    "not", "no", "without", "neither", "nor", "never", "none", "cannot",
    "cant", "lack", "lacking", "absence", "absent", "failed", "failure",
}

# ── Clinical vocabulary (identical to Component 3) ────────────────────────────
CLINICAL_VOCAB: list[str] = [
    # Small-molecule drugs
    "hydroxychloroquine", "azithromycin", "dexamethasone", "remdesivir",
    "ivermectin", "metformin", "atorvastatin", "lisinopril", "amlodipine",
    "omeprazole", "gabapentin", "sertraline", "quetiapine", "aripiprazole",
    "olanzapine", "clozapine", "haloperidol", "risperidone", "clopidogrel",
    "warfarin", "enoxaparin", "heparin", "acetaminophen", "ibuprofen",
    "methylprednisolone", "prednisone", "prednisolone", "cyclophosphamide",
    "methotrexate", "azathioprine", "tacrolimus", "cyclosporine",
    "mycophenolate", "sirolimus", "everolimus", "tamoxifen", "letrozole",
    "anastrozole", "exemestane", "erlotinib", "gefitinib", "imatinib",
    "dasatinib", "nilotinib", "sunitinib", "sorafenib", "vemurafenib",
    "dabrafenib", "trametinib", "osimertinib", "alectinib", "crizotinib",
    "palbociclib", "ribociclib", "abemaciclib", "olaparib", "niraparib",
    "rucaparib", "lenalidomide", "thalidomide", "pomalidomide",
    "bortezomib", "carfilzomib", "ixazomib", "venetoclax",
    # Biologics / monoclonal antibodies
    "bevacizumab", "cetuximab", "trastuzumab", "pertuzumab", "rituximab",
    "obinutuzumab", "ofatumumab", "alemtuzumab", "natalizumab",
    "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
    "ipilimumab", "avelumab", "cemiplimab", "tocilizumab", "sarilumab",
    "secukinumab", "ixekizumab", "guselkumab", "risankizumab",
    "ustekinumab", "belimumab", "eculizumab", "ravulizumab",
    "denosumab", "romosozumab", "mepolizumab", "dupilumab",
    "infliximab", "adalimumab", "etanercept", "golimumab", "certolizumab",
    "abatacept", "anakinra", "canakinumab", "basiliximab", "daclizumab",
    # Pharmacology / study-design terms
    "pharmacokinetics", "pharmacodynamics", "bioavailability",
    "immunosuppression", "immunosuppressive", "immunotherapy",
    "chemotherapy", "radiotherapy", "monotherapy", "polypharmacy",
    "bioequivalence", "dose-escalation", "dose-response",
    # Adverse-event / toxicity terms
    "nephrotoxicity", "hepatotoxicity", "cardiotoxicity", "neurotoxicity",
    "thrombocytopenia", "neutropenia", "leukopenia", "lymphopenia",
    "hyperglycemia", "hypoglycemia", "hyperlipidemia", "dyslipidemia",
    "thromboembolism", "thrombosis", "anticoagulation",
    "cardiomyopathy", "arrhythmia", "tachycardia", "bradycardia",
    "glomerulonephritis", "nephropathy", "neuropathy", "encephalopathy",
    # Multi-word clinical terms
    "myocardial infarction", "heart failure", "atrial fibrillation",
    "type 2 diabetes", "non-small cell", "randomized controlled",
    "immune checkpoint", "checkpoint inhibitor", "chimeric antigen",
    "overall survival", "progression-free survival", "adverse event",
    # Cellular / molecular biology
    "angiogenesis", "apoptosis", "proliferation", "differentiation",
    "cytokine", "interleukin", "interferon", "immunoglobulin",
    "lymphocyte", "neutrophil", "eosinophil", "basophil",
    "erythrocyte", "leukocyte", "fibrinogen", "creatinine", "hemoglobin",
    # Routes of administration
    "subcutaneous", "intravenous", "intramuscular", "intrathecal",
    "intraperitoneal", "transdermal",
]

# ── Tokenizer model IDs (identical to Component 3) ───────────────────────────
TOKENIZER_IDS = {
    "bert-base-uncased": "bert-base-uncased",
    "roberta-base":      "roberta-base",
    "clinicalbert":      "medicalai/ClinicalBERT",
}

FEATURE_COLS = [
    "summary_length_chars",
    "summary_length_words",
    "hedging_count",
    "negation_count",
    "hedging_density",
    "negation_density",
]


# ── MIMIC loader ──────────────────────────────────────────────────────────────

def load_mimic_radiology(noteevents_path: Path) -> pd.DataFrame:
    """Read NOTEEVENTS.csv in 10k-row chunks, filter to Radiology, sample 500 notes."""
    print(f"[INFO] Reading {noteevents_path} in chunks of {CHUNK_SIZE:,} rows …")

    radiology_chunks: list[pd.DataFrame] = []
    total_rows = 0

    reader = pd.read_csv(
        noteevents_path,
        usecols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CATEGORY", "TEXT"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )
    for chunk in reader:
        total_rows += len(chunk)
        radiology_chunk = chunk[chunk["CATEGORY"].str.strip().str.lower() == "radiology"]
        if len(radiology_chunk) > 0:
            radiology_chunks.append(radiology_chunk)

    print(f"[INFO] Total notes scanned: {total_rows:,}")

    radiology = pd.concat(radiology_chunks, ignore_index=True) if radiology_chunks else pd.DataFrame()
    print(f"[INFO] Radiology notes: {len(radiology):,}")

    # Drop rows with missing text
    radiology = radiology.dropna(subset=["TEXT"])
    radiology = radiology[radiology["TEXT"].str.strip().str.len() > 0]

    sample = radiology.sample(n=min(SAMPLE_N, len(radiology)), random_state=RANDOM_STATE)
    sample = sample.reset_index(drop=True)
    print(f"[INFO] Sampled {len(sample)} notes (random_state={RANDOM_STATE})\n")
    return sample


# ── Linguistic features (same logic as Component 2) ──────────────────────────

def word_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def count_matches(tokens: list[str], word_set: set[str]) -> int:
    return sum(1 for t in tokens if t in word_set)


def compute_linguistic_features(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Add all 6 linguistic feature columns using text_col as source."""
    tokens_series = df[text_col].apply(word_tokenize)

    df["summary_length_chars"] = df[text_col].str.len()
    df["summary_length_words"] = tokens_series.apply(len)
    df["hedging_count"]        = tokens_series.apply(lambda t: count_matches(t, HEDGING_WORDS))
    df["negation_count"]       = tokens_series.apply(lambda t: count_matches(t, NEGATION_WORDS))

    word_counts = df["summary_length_words"].replace(0, float("nan"))
    df["hedging_density"]  = df["hedging_count"]  / word_counts
    df["negation_density"] = df["negation_count"] / word_counts

    return df


# ── Tokenizer fragmentation (same logic as Component 3) ──────────────────────

def find_terms_in_text(text: str, vocab: list[str]) -> list[str]:
    text_lower = text.lower()
    return [
        term for term in vocab
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower)
    ]


def compute_fragmentation(term: str, tokenizer: AutoTokenizer) -> tuple[int, float]:
    tokens = tokenizer.tokenize(term)
    word_count = len(term.split())
    token_count = len(tokens)
    score = round(token_count / word_count, 4) if word_count > 0 else 0.0
    return token_count, score


def load_tokenizers(device: str) -> dict[str, AutoTokenizer]:
    tokenizers: dict[str, AutoTokenizer] = {}
    for name, model_id in TOKENIZER_IDS.items():
        print(f"[INFO] Loading tokenizer: {name} ({model_id}) [device={device}]")
        tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
    return tokenizers


def build_fragmentation_rows(
    df: pd.DataFrame,
    tokenizers: dict[str, AutoTokenizer],
    id_col: str,
) -> list[dict]:
    rows: list[dict] = []
    total = len(df)
    for i, record in enumerate(df.itertuples(index=False), start=1):
        if i % 100 == 0:
            print(f"  Fragmentation: row {i}/{total} …")
        text = getattr(record, "TEXT")
        row_id = getattr(record, id_col)
        terms = find_terms_in_text(text, CLINICAL_VOCAB)
        for term in terms:
            for tok_name, tokenizer in tokenizers.items():
                token_count, score = compute_fragmentation(term, tokenizer)
                rows.append({
                    "row_id":              row_id,
                    "term":                term,
                    "tokenizer":           tok_name,
                    "token_count":         token_count,
                    "fragmentation_score": score,
                })
    return rows


# ── Comparison summary ────────────────────────────────────────────────────────

def print_linguistic_comparison(mimic_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("LINGUISTIC FEATURES: MIMIC vs ClinicalTrials")
    print("=" * 70)

    trials_df = pd.read_csv(TRIALS_LING_PATH)
    trials_means = trials_df[FEATURE_COLS].mean().rename("ClinicalTrials (mean)")
    mimic_means  = mimic_df[FEATURE_COLS].mean().rename("MIMIC Radiology (mean)")

    comparison = pd.concat([trials_means, mimic_means], axis=1).round(4)
    comparison["delta"] = (comparison["MIMIC Radiology (mean)"] - comparison["ClinicalTrials (mean)"]).round(4)
    print(comparison.to_string())


def print_fragmentation_comparison(mimic_frag_df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TOKENIZER FRAGMENTATION: MIMIC vs ClinicalTrials")
    print("=" * 70)

    trials_frag_df = pd.read_csv(TRIALS_FRAG_PATH)

    trials_means = (
        trials_frag_df.groupby("tokenizer")["fragmentation_score"]
        .mean()
        .rename("ClinicalTrials (mean)")
    )
    mimic_means = (
        mimic_frag_df.groupby("tokenizer")["fragmentation_score"]
        .mean()
        .rename("MIMIC Radiology (mean)")
    )

    comparison = pd.concat([trials_means, mimic_means], axis=1).round(4)
    comparison["delta"] = (comparison["MIMIC Radiology (mean)"] - comparison["ClinicalTrials (mean)"]).round(4)
    print(comparison.to_string())

    print(f"\nMIMIC: unique terms matched across all notes: "
          f"{mimic_frag_df['term'].nunique()}")
    print(f"MIMIC: notes with at least one matched term: "
          f"{mimic_frag_df['row_id'].nunique()}")

    print("\nTop 10 most-fragmented MIMIC terms (mean score across tokenizers):")
    top = (
        mimic_frag_df.groupby("term")["fragmentation_score"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .round(4)
    )
    print(top.to_string())
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Component 4: MIMIC-III radiology analysis")
    parser.add_argument(
        "--noteevents",
        type=Path,
        default=DATA_DIR / "raw" / "NOTEEVENTS.csv",
        help="Path to MIMIC-III NOTEEVENTS.csv (default: data/raw/NOTEEVENTS.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.noteevents.exists():
        print(f"[ERROR] NOTEEVENTS.csv not found at {args.noteevents}")
        print("        Pass the correct path with --noteevents /path/to/NOTEEVENTS.csv")
        sys.exit(1)

    print(f"[INFO] Using device: {DEVICE}")

    # ── Step 1: Load and sample radiology notes ───────────────────────────────
    notes = load_mimic_radiology(args.noteevents)

    # ── Step 2: Linguistic features ───────────────────────────────────────────
    print("[INFO] Computing linguistic features …")
    notes = compute_linguistic_features(notes, text_col="TEXT")

    ling_out_cols = ["ROW_ID", "SUBJECT_ID", "HADM_ID"] + FEATURE_COLS
    notes[ling_out_cols].to_csv(MIMIC_LING_OUT, index=False)
    print(f"[SAVED] {MIMIC_LING_OUT}")

    # ── Step 3: Tokenizer fragmentation ──────────────────────────────────────
    print("\n[INFO] Loading tokenizers …")
    tokenizers = load_tokenizers(DEVICE)

    print(f"\n[INFO] Scanning {len(notes)} notes for {len(CLINICAL_VOCAB)} vocabulary terms …")
    frag_rows = build_fragmentation_rows(notes, tokenizers, id_col="ROW_ID")

    frag_df = pd.DataFrame(frag_rows)
    frag_df.to_csv(MIMIC_FRAG_OUT, index=False)
    print(f"[SAVED] {MIMIC_FRAG_OUT}")

    # ── Step 4: Comparison summaries ─────────────────────────────────────────
    print_linguistic_comparison(notes)
    print_fragmentation_comparison(frag_df)
