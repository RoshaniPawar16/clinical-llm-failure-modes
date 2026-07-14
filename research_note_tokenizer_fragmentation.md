# Tokenizer Fragmentation of Clinical Drug Terminology: A Comparative Analysis of BERT, RoBERTa, and ClinicalBERT

**Roshani Pawar**  
Centre for Healthcare Randomised Trials (CHaRT), University of Aberdeen  
Independent pre-PhD research | June 2026  
Repository: https://github.com/RoshaniPawar16/clinical-llm-failure-modes

---

## Background

Large language models (LLMs) process text by splitting words into subword tokens. For general English, this works well. For clinical and pharmacological terminology, it is a known failure mode: long, morphologically complex drug names are fragmented into meaningless subword pieces, destroying the semantic unit before the model ever processes it.

Domain-specific models such as ClinicalBERT were trained on clinical text precisely to address this problem. The implicit assumption is that exposure to clinical vocabulary during pre-training reduces fragmentation of clinical terms. This note tests that assumption directly.

---

## Method

500 completed interventional clinical trials were retrieved from the ClinicalTrials.gov REST API v2, filtered to Drug or Biological intervention types. From each trial's `brief_summary` field, clinical and pharmacological terms were matched against a curated vocabulary of 30 domain-specific terms including drug names, disease terms, and pharmacological concepts.

Each matched term was tokenized using three models:

- `bert-base-uncased` (general-purpose)
- `roberta-base` (general-purpose)
- `medicalai/ClinicalBERT` (domain-specific, trained on MIMIC-III clinical notes)

Fragmentation score was defined as the number of subword tokens produced for a given term divided by its word count. A score of 1.0 indicates no fragmentation; higher scores indicate greater splitting.

325 term-trial pairs were matched per tokenizer, covering 92 unique clinical terms across 201 trials.

---

## Results

| Tokenizer | Mean Fragmentation | Std | Max |
|---|---|---|---|
| bert-base-uncased | 3.71 | 1.72 | 6.0 |
| roberta-base | 3.53 | 1.17 | 7.0 |
| **ClinicalBERT** | **3.81** | **1.27** | **7.0** |

The most fragmented terms were:

| Term | Score | Tokenizer |
|---|---|---|
| cyclophosphamide | 7.0 | ClinicalBERT |
| thrombocytopenia | 7.0 | ClinicalBERT |
| pharmacokinetics | 6.0 | bert-base-uncased |
| dexamethasone | 6.0 | ClinicalBERT |
| bevacizumab | 6.0 | ClinicalBERT |

---

## Key Finding

**ClinicalBERT produces higher mean fragmentation (3.81) than general-purpose BERT (3.71) and RoBERTa (3.53).** This is counterintuitive. A model trained on clinical notes from MIMIC-III would be expected to develop a vocabulary that better accommodates clinical terminology. Instead, it fragments drug names more aggressively than the general-purpose alternatives.

This suggests that pre-training on clinical text does not automatically resolve tokenizer-level fragmentation. The subword vocabulary is fixed at tokenizer construction time, prior to pre-training, and clinical exposure during pre-training does not retroactively improve it. The result is a model that has learned clinical context but still fragments the very terms it most needs to represent as coherent units.

---

## Implications

A fragmentation score of 7 means a single drug name — cyclophosphamide, thrombocytopenia — becomes 7 separate tokens. Attention mechanisms must reconstruct meaning across those 7 pieces with no guarantee of coherence. For downstream tasks such as named entity recognition, relation extraction, or clinical summarisation, this is a structural disadvantage that affects ClinicalBERT more, not less, than its general-purpose counterparts.

This finding supports the argument that domain adaptation for clinical NLP requires tokenizer-level intervention — not just pre-training on clinical text — to meaningfully reduce fragmentation of pharmacological and disease terminology.

---

## Limitations

The term vocabulary used here is hardcoded and limited to 30 terms. Results may not generalise across the full range of clinical terminology. The fragmentation score metric is simple and does not account for whether fragmented subwords retain partial semantic content. MIMIC-III access is pending; future work will replicate this analysis on real radiology notes.

---

## Next Steps

- Replicate on MIMIC-III radiology notes once PhysioNet credentialing is approved
- Test vocabulary-expanded tokenizers (e.g. adding clinical terms to the base vocabulary)
- Cross-reference fragmentation scores with downstream NER performance on clinical benchmarks
