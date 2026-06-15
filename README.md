# clinical-llm-failure-modes

Pre-PhD research: building evidence for LLM failure modes in clinical NLP.

---

## Structure

```
clinical-llm-failure-modes/
├── component1_clinicaltrials/
│   └── fetch_trials.py          ← ClinicalTrials.gov downloader (no auth needed)
├── data/
│   ├── raw/                     ← full API output
│   └── processed/               ← clean dataframes (committed to git)
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
cd clinical-llm-failure-modes
pip install -r requirements.txt
python component1_clinicaltrials/fetch_trials.py
# Optional: specify how many rows
python component1_clinicaltrials/fetch_trials.py 1000
```

Output written to `data/processed/trials_clean.csv`.

---

## Roadmap

- [x] Component 1: ClinicalTrials.gov public API
- [ ] Component 2: MIMIC-III radiology notes (pending PhysioNet CITI approval)
- [ ] Component 3: Linguistic register analysis (hedging, negation, vocabulary shift)
- [ ] Component 4: LLM tokenizer fragmentation on clinical subgroups
