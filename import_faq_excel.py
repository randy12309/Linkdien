#!/usr/bin/env python
"""Import FAQ from Excel file into Valkey cache and update faq.json."""
import json
import sys
import uuid
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_server import cache as valkey_cache

DATA_DIR = Path(__file__).resolve().parents[1] / "app" / "data"
FAQ_JSON = DATA_DIR / "faq.json"
EXCEL_FILE = Path(__file__).resolve().parents[2] / "data" / "SAG_AFTRA_300_FAQ_Questions_updated.xlsx"

NAMESPACE = uuid.UUID("6f1d5c1e-52a0-4c58-8a43-6d1f0d6c7a11")


def _id(*parts: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "\x1f".join(parts)))


def _hash(*parts: str) -> str:
    import hashlib
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def normalize(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower().replace("'", ""))).strip()


def faq_hash(e: dict) -> str:
    return _hash(e["question"], e.get("answer") or "", e.get("source", "faq"))


def read_excel_faqs() -> list[dict]:
    """Read FAQ entries from Excel file."""
    df = pd.read_excel(EXCEL_FILE)
    faqs = []
    for _, row in df.iterrows():
        q_no = row["Question No."]
        topic = row["Topic"]
        question = str(row["Question"]).strip()
        answer = str(row["Answer"]).strip() if pd.notna(row["Answer"]) else None
        answer_type = str(row["Answer type"]).strip() if pd.notna(row["Answer type"]) else None
        policy_refs = str(row["Policy references"]).strip() if pd.notna(row["Policy references"]) else None

        # Create unique ID based on question number
        faq_id = f"xls-{q_no:03d}"

        # Build source info
        source_parts = ["faq"]
        if topic:
            source_parts.append(f"topic:{topic}")
        if answer_type:
            source_parts.append(f"type:{answer_type}")
        if policy_refs:
            source_parts.append(f"policies:{policy_refs}")
        source = "|".join(source_parts)

        faqs.append({
            "id": faq_id,
            "question": question,
            "answer": answer,
            "source": source,
        })
    return faqs


def load_existing_faq_json() -> list[dict]:
    """Load existing faq.json."""
    if FAQ_JSON.exists():
        return json.loads(FAQ_JSON.read_text(encoding="utf-8"))
    return []


def merge_faqs(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge new FAQs with existing, avoiding duplicates by question text."""
    existing_questions = {normalize(e["question"]): e for e in existing}
    merged = list(existing)

    for n in new:
        nq = normalize(n["question"])
        if nq not in existing_questions:
            merged.append(n)
            existing_questions[nq] = n
    return merged


def warm_valkey_cache(faqs: list[dict]) -> int:
    """Warm Valkey cache with FAQ entries."""
    try:
        client = valkey_cache._client()
        if client is None:
            print("Valkey not configured, skipping cache warm")
            return 0

        cached = 0
        pipe = client.pipeline()
        for e in faqs:
            if e.get("answer") is None:
                continue  # Skip member-specific questions (no answer)
            key = normalize(e["question"])
            value = json.dumps({
                "id": e["id"],
                "answer": e.get("answer"),
                "source": e.get("source", "faq"),
            })
            faq_key = valkey_cache._faq_key(key)
            pipe.set(faq_key, value, ex=valkey_cache.FAQ_TTL)
            cached += 1
        pipe.execute()
        return cached
    except Exception as exc:
        print(f"Valkey cache warm failed: {exc}")
        return 0


def main():
    print("Reading Excel file...")
    excel_faqs = read_excel_faqs()
    print(f"Found {len(excel_faqs)} FAQ entries in Excel")

    print("Loading existing faq.json...")
    existing_faqs = load_existing_faq_json()
    print(f"Existing FAQ entries: {len(existing_faqs)}")

    print("Merging FAQs...")
    merged_faqs = merge_faqs(existing_faqs, excel_faqs)
    print(f"Total FAQ entries after merge: {len(merged_faqs)}")

    # Save merged faq.json
    FAQ_JSON.write_text(json.dumps(merged_faqs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Updated {FAQ_JSON}")

    # Warm Valkey cache
    print("Warming Valkey cache...")
    cached = warm_valkey_cache(merged_faqs)
    print(f"Cached {cached} entries in Valkey")

    print("Done!")


if __name__ == "__main__":
    import uuid
    main()