"""Reversible pseudonym masking for sensitive columns.

Protects two boundaries in the pipeline:
  1. The enrichment agent's prompt (agent 2) — prevents exposing real
     sensitive values.
  2. Session memory — agent 1 reads memory_context back on the next turn, so
     whatever gets written to disk via session_memory.append_turn does not
     carry real values from those columns either.

A MaskKey is built per turn, scoped to whatever slice of data is actually
going to be exposed at that point. It is never persisted: the mapping only
lives in-process for the request that built it.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def secure_columns_for(schema: dict) -> dict[str, str]:
    """Returns {column_name: pseudonym_prefix} for every column flagged
    secure=true in the project schema (written to schema.json at confirm time).
    PK/FK columns use prefix "ID"; any other user-marked column uses "Value".
    Called by build_mask_key in place of the old name-fragment heuristic."""
    result: dict[str, str] = {}
    for table in schema.get("tables", []):
        for col in table.get("columns", []):
            if col.get("secure"):
                key_type = col.get("key_type", "")
                prefix = "ID" if key_type in ("PK", "FK") else "Value"
                result[col["name"]] = prefix
    return result


@dataclass
class MaskKey:
    """Per-turn mapping. key_id (user:session:mode:question:answer, or
    reply_to:<answerId> when reconstructed via from_storage) is for log
    traceability only — never parsed."""

    key_id: str
    forward: dict[str, dict[str, str]] = field(default_factory=dict)  # column -> {real: pseudonym}
    backward: dict[str, dict[str, str]] = field(default_factory=dict)  # column -> {pseudonym: real}

    def is_empty(self) -> bool:
        # Checks both directions, not just forward: mask_json/demask_json
        # each rely on only one of the two, and a key reconstructed by
        # from_storage from a partial/malformed record could plausibly have
        # one populated without the other.
        return not self.forward and not self.backward

    @property
    def columns(self) -> list[str]:
        return list(dict.fromkeys([*self.forward.keys(), *self.backward.keys()]))


def merge_keys(*keys: "MaskKey", key_id: str) -> "MaskKey":
    """Combines multiple MaskKeys' forward/backward maps into one, column by
    column. Needed when more than one masking concern applies to the same
    agent 1 call in the same turn — e.g. a pseudonym echoed back from
    memory_context (reply_to_mask_key) and a merchant name just resolved
    from a disambiguation selection (see
    query_orchestrator._mask_resolved_merchant_filter) both need demasking
    out of the same generated-code string before execution. Empty keys
    contribute nothing; on a same-column collision the later argument's
    entries win (shouldn't happen in practice — the keys are built from
    disjoint turns/values — but defined rather than left to silently
    misbehave)."""
    merged = MaskKey(key_id=key_id)
    for key in keys:
        for col, forward in key.forward.items():
            merged.forward.setdefault(col, {}).update(forward)
        for col, backward in key.backward.items():
            merged.backward.setdefault(col, {}).update(backward)
    return merged


def build_mask_key(
    *,
    user_id: str,
    session_id: str,
    mode: str,
    question_id: str,
    answer_id: str,
    df: pd.DataFrame,
    secure_cols: dict[str, str] | None = None,
) -> MaskKey:
    """Assigns each unique value in every secure column a stable pseudonym
    (ID1, Value2, …) in first-appearance order. `secure_cols` is
    {col_name: prefix} from secure_columns_for(schema) — only columns the
    user explicitly flagged secure (PK/FK auto-flagged at upload) are masked.
    Call this on the smallest dataframe that will actually be exposed
    downstream."""
    key = MaskKey(key_id=f"{user_id}:{session_id}:{mode}:{question_id}:{answer_id}")

    for col, prefix in (secure_cols or {}).items():
        if col not in df.columns:
            continue
        forward: dict[str, str] = {}
        backward: dict[str, str] = {}
        seq = 0
        for val in df[col]:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue
            real = str(val)
            if real in forward:
                continue
            seq += 1
            pseudonym = f"{prefix}{seq}"
            forward[real] = pseudonym
            backward[pseudonym] = real
        if forward:
            key.forward[col] = forward
            key.backward[col] = backward

    return key


def mask_dataframe(df: pd.DataFrame, key: MaskKey) -> pd.DataFrame:
    """Returns a copy of `df` with sensitive columns replaced by their
    pseudonyms. No-op (returns df unchanged) if key is empty."""
    if key.is_empty():
        return df
    masked = df.copy()
    for col, forward in key.forward.items():
        if col in masked.columns:
            masked[col] = masked[col].map(lambda v: forward.get(str(v), v) if pd.notna(v) else v)
    return masked


def _match_case(replacement: str, sample: str) -> str:
    """Mirrors `sample`'s casing pattern onto `replacement`. Generated pandas
    code frequently lowercases pseudonyms itself as part of a case-insensitive
    filter (`df[col].str.lower().isin([...])` with a literal lowercase list)
    — demasking "agent1" back to the canonically-cased real value would leave
    that comparison silently matching nothing, since the real column only
    gets lowered on one side. Mirroring the matched token's case keeps
    whatever comparison the code is doing working after the swap."""
    if sample.islower():
        return replacement.lower()
    if sample.isupper():
        return replacement.upper()
    return replacement


def _replace_tokens(text: str, mapping: dict[str, str]) -> str:
    """Case-insensitive find/replace of every key in `mapping`, longest first
    so e.g. "Merchant1" can't win a match that should have been "Merchant10"
    (alternation tries patterns left-to-right at each position, so sorting
    longest-first makes the longer candidate claim the match before the
    shorter one is tried). The replacement's case mirrors whatever case the
    matched text was actually found in — see _match_case."""
    if not mapping:
        return text
    lower_lookup = {k.lower(): v for k, v in mapping.items()}
    pattern = re.compile(
        "|".join(re.escape(k) for k in sorted(mapping, key=len, reverse=True)), re.IGNORECASE
    )
    return pattern.sub(lambda m: _match_case(lower_lookup[m.group(0).lower()], m.group(0)), text)


def _walk(obj: Any, column_maps: dict[str, dict[str, str]]) -> Any:
    """Recursively applies every column's mapping to every string found in
    obj, regardless of nesting. Needed because the enrichment agent's
    decision payload and the response envelope don't tag values by source
    column — a pseudonym like "Merchant3" is just a string wherever it shows
    up (table cell, chart label, narrative sentence)."""
    if isinstance(obj, str):
        text = obj
        for mapping in column_maps.values():
            text = _replace_tokens(text, mapping)
        return text
    if isinstance(obj, dict):
        return {k: _walk(v, column_maps) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v, column_maps) for v in obj]
    return obj


def mask_json(obj: Any, key: MaskKey) -> Any:
    """Replaces real values with pseudonyms anywhere they appear in obj."""
    if key.is_empty():
        return obj
    return _walk(obj, key.forward)


def demask_json(obj: Any, key: MaskKey) -> Any:
    """Replaces pseudonyms with real values anywhere they appear in obj.
    Works on a plain string too (e.g. agent-1-generated pandas code), not
    just dict/list JSON structures — see _walk."""
    if key.is_empty():
        return obj
    return _walk(obj, key.backward)


def to_storage(key: MaskKey) -> dict:
    """Serializes a MaskKey's mapping (not its key_id) for persistence
    alongside the turn it masked, in session_memory.append_turn — so a later
    turn replying to this one (via its incoming answerId) can look the
    mapping back up by answer_id and demask newly-generated code before it
    runs against the real dataframe."""
    return {"forward": key.forward, "backward": key.backward}


def from_storage(data: dict, key_id: str = "") -> MaskKey:
    """Reconstructs a MaskKey from what to_storage() persisted."""
    return MaskKey(key_id=key_id, forward=data.get("forward") or {}, backward=data.get("backward") or {})