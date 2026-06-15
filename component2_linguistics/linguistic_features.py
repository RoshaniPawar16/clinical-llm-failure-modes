"""
Component 2: Linguistic Feature Extraction
===========================================
Computes hedging and negation features on clinical trial brief summaries.

Input:  data/processed/trials_clean.csv
Output: data/processed/linguistic_features.csv
Columns added: summary_length_chars | summary_length_words | hedging_count |
               negation_count | hedging_density | negation_density
"""

import re
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "processed" / "trials_clean.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "linguistic_features.csv"

# ── Word lists ────────────────────────────────────────────────────────────────
HEDGING_WORDS = {
    "possible", "possibly", "may", "might", "could", "suggest", "appears",
    "seems", "potential", "potentially", "unclear", "uncertain", "likely",
    "unlikely", "whether", "approximately", "evaluate", "assess",
}

NEGATION_WORDS = {
    "not", "no", "without", "neither", "nor", "never", "none", "cannot",
    "cant", "lack", "lacking", "absence", "absent", "failed", "failure",
}


# ── Feature computation ───────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return re.findall(r"[a-z']+", text.lower())


def count_matches(tokens: list[str], word_set: set[str]) -> int:
    return sum(1 for t in tokens if t in word_set)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all 6 linguistic feature columns to df in place and return it."""
    tokens_series = df["brief_summary"].apply(tokenize)

    df["summary_length_chars"] = df["brief_summary"].str.len()
    df["summary_length_words"] = tokens_series.apply(len)
    df["hedging_count"]        = tokens_series.apply(lambda t: count_matches(t, HEDGING_WORDS))
    df["negation_count"]       = tokens_series.apply(lambda t: count_matches(t, NEGATION_WORDS))

    # Guard against zero-length summaries to avoid division by zero
    word_counts = df["summary_length_words"].replace(0, float("nan"))
    df["hedging_density"]  = df["hedging_count"]  / word_counts
    df["negation_density"] = df["negation_count"] / word_counts

    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "summary_length_chars",
    "summary_length_words",
    "hedging_count",
    "negation_count",
    "hedging_density",
    "negation_density",
]


def print_grouped_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("GROUPED MEANS BY INTERVENTION TYPE")
    print("=" * 70)
    grouped = df.groupby("intervention_type")[FEATURE_COLS].mean().round(4)
    print(grouped.to_string())


def print_overall_stats(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("OVERALL DESCRIPTIVE STATS (all 500 rows)")
    print("=" * 70)
    print(df[FEATURE_COLS].describe().round(4).to_string())
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[INFO] Loading {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {len(df)} rows")

    df = compute_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["nct_id", "brief_summary", "intervention_type"] + FEATURE_COLS].to_csv(
        OUTPUT_PATH, index=False
    )
    print(f"[SAVED] {OUTPUT_PATH}")

    print_grouped_summary(df)
    print_overall_stats(df)
