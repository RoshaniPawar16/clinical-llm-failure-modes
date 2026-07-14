# clinical-llm-failure-modes

Pre-PhD research: building evidence for LLM failure modes in clinical NLP.

---

## Structure

```
clinical-llm-failure-modes/
├── component1_clinicaltrials/
│   └── fetch_trials.py              ← ClinicalTrials.gov downloader (no auth needed)
├── component2_linguistics/
│   └── linguistic_features.py       ← Hedging/negation feature extraction
├── component3_tokenizer/
│   └── tokenizer_fragmentation.py   ← Tokenizer fragmentation analysis
├── component4_mimic/
│   └── mimic_analysis.py            ← MIMIC-III radiology notes analysis
├── data/
│   ├── raw/                         ← full API output (gitignored)
│   └── processed/
│       ├── trials_clean.csv         ← 500 completed interventional trials
│       ├── linguistic_features.csv  ← trials_clean + 6 linguistic feature cols
│       ├── tokenizer_fragmentation.csv    ← per-term fragmentation scores (ClinicalTrials)
│       ├── mimic_linguistic_features.csv  ← linguistic features on MIMIC radiology notes
│       └── mimic_tokenizer_fragmentation.csv ← fragmentation scores on MIMIC notes
├── requirements.txt
└── README.md
```

---

## Component 1 — ClinicalTrials.gov Protocol Summaries

**What it does:** Downloads completed interventional trials with Drug or Biological interventions from the public ClinicalTrials.gov API v2. No account required.

**Output columns:**

| Column | Description |
|---|---|
| `nct_id` | ClinicalTrials.gov identifier (e.g. NCT01234567) |
| `brief_summary` | Plain-English protocol summary text |
| `intervention_type` | `DRUG` or `BIOLOGICAL` |

**Run it:**

```bash
pip install -r requirements.txt
python component1_clinicaltrials/fetch_trials.py
# Optional: specify how many rows
python component1_clinicaltrials/fetch_trials.py 1000
```

Output: `data/processed/trials_clean.csv`

---

## Component 2 — Linguistic Feature Extraction

**What it does:** Computes hedging and negation features on each trial's `brief_summary`. Uses only the standard library and pandas — no NLP models required.

**Features computed:**

| Column | Description |
|---|---|
| `summary_length_chars` | Character count of brief_summary |
| `summary_length_words` | Word count (whitespace-tokenised) |
| `hedging_count` | Occurrences of hedging words (possible, may, might, suggest, …) |
| `negation_count` | Occurrences of negation words (not, no, without, never, …) |
| `hedging_density` | `hedging_count / summary_length_words` |
| `negation_density` | `negation_count / summary_length_words` |

**Run it:**

```bash
python component2_linguistics/linguistic_features.py
```

Output: `data/processed/linguistic_features.csv`

---

## Component 3 — Tokenizer Fragmentation Analysis

**What it does:** For each trial summary, finds clinical and drug terms from a hardcoded vocabulary and measures how three tokenizers fragment them. Quantifies subword fragmentation — a known failure mode where domain-specific terminology is split into meaningless pieces.

**Tokenizers tested:** `bert-base-uncased`, `roberta-base`, `medicalai/ClinicalBERT`

**Output columns:**

| Column | Description |
|---|---|
| `nct_id` | Trial identifier |
| `intervention_type` | `DRUG` or `BIOLOGICAL` |
| `term` | Clinical/drug term matched in the summary |
| `tokenizer` | Tokenizer name |
| `token_count` | Number of subword tokens the term was split into |
| `fragmentation_score` | `token_count / word_count` — higher means more fragmentation |

**Run it:**

```bash
pip install transformers
python component3_tokenizer/tokenizer_fragmentation.py
```

Output: `data/processed/tokenizer_fragmentation.csv`

---

## Component 4 — MIMIC-III Radiology Notes

**What it does:** Applies the same linguistic feature extraction (Component 2) and
tokenizer fragmentation analysis (Component 3) to 500 sampled MIMIC-III radiology
notes, then prints a side-by-side comparison against the ClinicalTrials.gov baseline.
Requires PhysioNet credentialed access to MIMIC-III.

**Output files:**

| File | Description |
|---|---|
| `mimic_linguistic_features.csv` | ROW_ID, SUBJECT_ID, HADM_ID + 6 linguistic feature cols |
| `mimic_tokenizer_fragmentation.csv` | row_id, term, tokenizer, token_count, fragmentation_score |

**Run it:**

```bash
# Default path (place NOTEEVENTS.csv in data/raw/)
python component4_mimic/mimic_analysis.py

# Custom path
python component4_mimic/mimic_analysis.py --noteevents /path/to/NOTEEVENTS.csv
```

---

## Roadmap

- [x] Component 1: ClinicalTrials.gov public API
- [x] Component 2: Linguistic feature extraction (hedging, negation)
- [x] Component 3: Tokenizer fragmentation analysis
- [x] Component 4: MIMIC-III radiology notes analysis
