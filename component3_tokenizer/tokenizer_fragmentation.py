"""
Component 3: Tokenizer Fragmentation Analysis
=============================================
For each trial summary, finds clinical/drug terms from a hardcoded vocabulary
and measures how three tokenizers fragment them into subword tokens.

Fragmentation score = token_count / word_count
  - Score of 1.0 → no fragmentation (each word = 1 token)
  - Score > 1.0 → fragmentation (term split into subword pieces)

Input:  data/processed/trials_clean.csv
Output: data/processed/tokenizer_fragmentation.csv
Columns: nct_id | intervention_type | term | tokenizer | token_count | fragmentation_score
"""

import re
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_PATH = DATA_DIR / "processed" / "trials_clean.csv"
OUTPUT_PATH = DATA_DIR / "processed" / "tokenizer_fragmentation.csv"

# ── Tokenizer model IDs ───────────────────────────────────────────────────────
TOKENIZER_IDS = {
    "bert-base-uncased":  "bert-base-uncased",
    "roberta-base":       "roberta-base",
    "clinicalbert":       "medicalai/ClinicalBERT",
}

# ── Clinical / drug vocabulary ────────────────────────────────────────────────
# Terms likely to appear in ClinicalTrials summaries; mix of single- and
# multi-word entries so fragmentation_score = token_count / word_count.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_terms_in_text(text: str, vocab: list[str]) -> list[str]:
    """Return vocab terms present in text (case-insensitive, whole-word match)."""
    text_lower = text.lower()
    return [
        term for term in vocab
        if re.search(r"\b" + re.escape(term) + r"\b", text_lower)
    ]


def compute_fragmentation(term: str, tokenizer: AutoTokenizer) -> tuple[int, float]:
    """
    Tokenize term and return (token_count, fragmentation_score).
    fragmentation_score = token_count / word_count
    """
    tokens = tokenizer.tokenize(term)
    word_count = len(term.split())
    token_count = len(tokens)
    score = round(token_count / word_count, 4) if word_count > 0 else 0.0
    return token_count, score


# ── Core pipeline ─────────────────────────────────────────────────────────────

def load_tokenizers() -> dict[str, AutoTokenizer]:
    tokenizers: dict[str, AutoTokenizer] = {}
    for name, model_id in TOKENIZER_IDS.items():
        print(f"[INFO] Loading tokenizer: {name} ({model_id})")
        tokenizers[name] = AutoTokenizer.from_pretrained(model_id)
    return tokenizers


def build_fragmentation_rows(
    df: pd.DataFrame,
    tokenizers: dict[str, AutoTokenizer],
) -> list[dict]:
    rows: list[dict] = []
    total = len(df)

    for i, record in enumerate(df.itertuples(index=False), start=1):
        if i % 100 == 0:
            print(f"  Processing row {i}/{total} …")

        terms = find_terms_in_text(record.brief_summary, CLINICAL_VOCAB)
        for term in terms:
            for tok_name, tokenizer in tokenizers.items():
                token_count, score = compute_fragmentation(term, tokenizer)
                rows.append({
                    "nct_id":               record.nct_id,
                    "intervention_type":    record.intervention_type,
                    "term":                 term,
                    "tokenizer":            tok_name,
                    "token_count":          token_count,
                    "fragmentation_score":  score,
                })

    return rows


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("FRAGMENTATION SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(df):,}  "
          f"({df['nct_id'].nunique()} trials, "
          f"{df['term'].nunique()} unique terms)")

    print("\nMean fragmentation_score by tokenizer:")
    print(
        df.groupby("tokenizer")["fragmentation_score"]
        .mean().round(4).to_string()
    )

    print("\nMean fragmentation_score by tokenizer × intervention_type:")
    print(
        df.groupby(["tokenizer", "intervention_type"])["fragmentation_score"]
        .mean().round(4).to_string()
    )

    print("\nTop 15 most-fragmented terms (mean score across all tokenizers):")
    top = (
        df.groupby("term")["fragmentation_score"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .round(4)
    )
    print(top.to_string())

    print("\nOverall stats on fragmentation_score:")
    print(df["fragmentation_score"].describe().round(4).to_string())
    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[INFO] Loading {INPUT_PATH}")
    trials = pd.read_csv(INPUT_PATH)
    print(f"[INFO] Loaded {len(trials)} rows\n")

    tokenizers = load_tokenizers()
    print(f"\n[INFO] Scanning {len(trials)} summaries for {len(CLINICAL_VOCAB)} vocabulary terms …\n")

    rows = build_fragmentation_rows(trials, tokenizers)

    result = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[SAVED] {OUTPUT_PATH}")

    print_summary(result)
