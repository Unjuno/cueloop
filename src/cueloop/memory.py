from __future__ import annotations

import json
from pathlib import Path

from .personalization import PersonalizationProfile, RetainedStructure
from .session import verify_session


STRUCTURE_COLLECTIONS = (
    "claims",
    "premises",
    "evidence",
    "counterexamples",
    "unresolved_questions",
    "verification_queue",
)


def retain_session_structure(
    session_directory: Path,
    item_id: str,
    profile_path: Path,
    *,
    retention_days: int = 365,
) -> RetainedStructure:
    errors = verify_session(session_directory)
    if errors:
        raise ValueError(f"session integrity verification failed: {errors}")
    manifest = json.loads((session_directory / "manifest.json").read_text(encoding="utf-8"))
    structure = json.loads((session_directory / "structure.json").read_text(encoding="utf-8"))
    selected: tuple[str, dict[str, object]] | None = None
    for collection in STRUCTURE_COLLECTIONS:
        values = structure.get(collection, [])
        if not isinstance(values, list):
            raise ValueError(f"structure collection {collection} is invalid")
        for item in values:
            if isinstance(item, dict) and item.get("id") == item_id:
                selected = (collection, item)
                break
        if selected is not None:
            break
    if selected is None:
        raise ValueError(f"structure item not found: {item_id}")
    collection, item = selected
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("structure item has no retainable text")
    profile = PersonalizationProfile.load(profile_path)
    retained = profile.retain_structure(
        kind=collection,
        text=text,
        source_session_id=manifest["session_id"],
        retention_days=retention_days,
    )
    profile.save(profile_path)
    return retained

