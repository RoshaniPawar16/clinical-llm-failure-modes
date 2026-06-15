"""
Component 1: ClinicalTrials.gov Protocol Summaries
====================================================
Downloads and parses interventional trials with Drug or Biological interventions.
Uses the ClinicalTrials.gov REST API v2 (no account needed, public).

Output: data/processed/trials_clean.csv
Columns: nct_id | brief_summary | intervention_type
"""

import requests
import pandas as pd
import time
import sys
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE = "https://clinicaltrials.gov/api/v2/studies"

# Filter: interventional trials only, drug or biological interventions
PARAMS = {
    "filter.overallStatus": "COMPLETED",       # completed = richer summaries
    "filter.advanced": "AREA[StudyType]INTERVENTIONAL",  # v2 API syntax for study type
    "fields": (
        "NCTId,"
        "BriefSummary,"
        "InterventionType,"
        "InterventionName,"
        "OfficialTitle,"
        "Phase,"
        "EnrollmentCount"
    ),
    "pageSize": 100,                           # max per page
    "countTotal": "true",
}

VALID_INTERVENTION_TYPES = {"DRUG", "BIOLOGICAL"}
TARGET_ROWS = 500                              # how many clean rows to collect
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_field(study: dict, *keys: str) -> str:
    """Safely drill into nested dict. Returns '' if any key is missing."""
    node = study
    for k in keys:
        if not isinstance(node, dict):
            return ""
        node = node.get(k, "")
    return node if isinstance(node, str) else ""


def extract_intervention_types(study: dict) -> set[str]:
    """Return set of intervention type strings for this study (uppercased)."""
    arms = (
        study
        .get("protocolSection", {})
        .get("armsInterventionsModule", {})
        .get("interventions", [])
    )
    types = set()
    for arm in arms:
        t = arm.get("type", "").upper()
        if t:
            types.add(t)
    return types


def extract_primary_intervention_type(study: dict) -> str:
    """
    Return the single most relevant intervention type label.
    Priority: DRUG > BIOLOGICAL > (whatever else matched)
    """
    types = extract_intervention_types(study) & VALID_INTERVENTION_TYPES
    if "DRUG" in types:
        return "DRUG"
    if "BIOLOGICAL" in types:
        return "BIOLOGICAL"
    return ""


def parse_study(study: dict) -> dict | None:
    """
    Parse one study dict from the API response.
    Returns a clean row dict or None if it doesn't meet criteria.
    """
    proto = study.get("protocolSection", {})

    # ── NCT ID ──
    nct_id = extract_field(proto, "identificationModule", "nctId")
    if not nct_id:
        return None

    # ── Brief summary ──
    brief_summary = extract_field(proto, "descriptionModule", "briefSummary")
    if not brief_summary or len(brief_summary.strip()) < 50:
        # Skip stubs — useless for LLM analysis
        return None

    # ── Intervention type filter ──
    intervention_type = extract_primary_intervention_type(study)
    if not intervention_type:
        return None

    return {
        "nct_id": nct_id,
        "brief_summary": brief_summary.strip().replace("\n", " "),
        "intervention_type": intervention_type,
    }


# ── Main download loop ────────────────────────────────────────────────────────

def fetch_trials(target_rows: int = TARGET_ROWS) -> pd.DataFrame:
    rows = []
    next_page_token = None
    page = 0

    print(f"[INFO] Target: {target_rows} clean rows (DRUG or BIOLOGICAL, INTERVENTIONAL, COMPLETED)")
    print(f"[INFO] API: {API_BASE}\n")

    while len(rows) < target_rows:
        params = {**PARAMS}
        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            resp = requests.get(
                API_BASE,
                params=params,
                timeout=30,
                headers={"User-Agent": "clinical-llm-failure-modes/0.1 (research; github.com/RoshaniPawar16/clinical-llm-failure-modes)"},
            )
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed on page {page}: {e}")
            break

        data = resp.json()
        studies = data.get("studies", [])
        total = data.get("totalCount", "?")
        next_page_token = data.get("nextPageToken")

        page += 1
        accepted = 0
        for study in studies:
            row = parse_study(study)
            if row:
                rows.append(row)
                accepted += 1

        print(
            f"  Page {page:>3} | fetched {len(studies):>3} | "
            f"accepted {accepted:>3} | total collected {len(rows):>4} / {target_rows} "
            f"(API total: {total})"
        )

        if not next_page_token:
            print("[INFO] No more pages from API.")
            break

        if len(rows) >= target_rows:
            break

        time.sleep(0.3)   # be polite to the API

    df = pd.DataFrame(rows[:target_rows])
    return df


# ── Save ─────────────────────────────────────────────────────────────────────

def save(df: pd.DataFrame) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / "trials_raw.csv"
    clean_path = OUT_DIR / "trials_clean.csv"

    df.to_csv(raw_path, index=False)
    df[["nct_id", "brief_summary", "intervention_type"]].to_csv(clean_path, index=False)

    print(f"\n[SAVED] Raw → {raw_path}")
    print(f"[SAVED] Clean → {clean_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATAFRAME SUMMARY")
    print("=" * 60)
    print(f"Shape:           {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Columns:         {list(df.columns)}")
    print(f"\nIntervention type distribution:")
    print(df["intervention_type"].value_counts().to_string())
    print(f"\nMissing values:")
    print(df.isnull().sum().to_string())
    print(f"\nSummary length stats (chars):")
    print(df["brief_summary"].str.len().describe().to_string())
    print("\nSample rows:")
    print(df.head(3).to_string(max_colwidth=80))
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_ROWS
    df = fetch_trials(target_rows=target)

    if df.empty:
        print("[ERROR] No data collected. Check your internet connection or API status.")
        sys.exit(1)

    save(df)
    print_summary(df)
