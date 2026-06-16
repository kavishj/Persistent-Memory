"""
core/retrieval/context_assembler.py

Packs reranked memories into the token budget with hard slot reservation.

Token budget (spec Day 2 — DO NOT CHANGE):
  Target:           1500 tokens
  Hard ceiling:     2000 tokens
  Procedural slot:   600 tokens (hard reserve — cannot be consumed by semantic)
  Semantic slot:     600 tokens (or 1000 if no procedure found)
  Episodic slot:     300 tokens (or 500 if no procedure found)

Rules:
  - Procedural slot is HARD RESERVE. Semantic cannot consume it.
  - Never truncate mid-sentence.
  - Assemble at sentence boundaries.
  - If no procedural found, redistribute: +400 to semantic, +200 to episodic.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from core.retrieval.query_builder import RawMemory, QueryResult


# ---------------------------------------------------------------------------
# Token budget constants (spec Day 2)
# ---------------------------------------------------------------------------
BUDGET_TARGET          = 1500
BUDGET_HARD_CEILING    = 2000

SLOT_PROCEDURAL        = 600   # hard reserve
SLOT_SEMANTIC_BASE     = 600
SLOT_EPISODIC_BASE     = 300

SLOT_SEMANTIC_NO_PROC  = 1000  # semantic gets extra if no procedure
SLOT_EPISODIC_NO_PROC  = 500   # episodic gets extra if no procedure


# ---------------------------------------------------------------------------
# Rough token estimator (no tiktoken dependency — 1 token ≈ 4 chars)
# ---------------------------------------------------------------------------
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Sentence-boundary truncation
# ---------------------------------------------------------------------------
def _truncate_to_sentences(text: str, max_tokens: int) -> str:
    """
    Truncates text to fit within max_tokens, stopping at the last
    complete sentence boundary. Never cuts mid-sentence.
    """
    if _estimate_tokens(text) <= max_tokens:
        return text

    # Split on sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    result = []
    used = 0
    for sentence in sentences:
        cost = _estimate_tokens(sentence + " ")
        if used + cost > max_tokens:
            break
        result.append(sentence)
        used += cost

    return " ".join(result) if result else text[:max_tokens * 4]


# ---------------------------------------------------------------------------
# Procedural memory serializer
# ---------------------------------------------------------------------------
def _serialize_procedural(memory: RawMemory, max_tokens: int) -> str:
    """
    Serializes a procedural memory into readable prompt text.
    Uses pre-computed summary if full procedure exceeds max_tokens.
    """
    props = memory.properties
    detail = props.get("detail", {})
    if isinstance(detail, str):
        try:
            detail = json.loads(detail)
        except Exception:
            detail = {}

    trigger   = props.get("trigger_condition", memory.content)
    task_type = props.get("task_type", "")
    steps     = detail.get("steps", [])
    edge_cases = detail.get("edge_cases", [])
    expected  = detail.get("expected_outcome", "")
    summary   = detail.get("summary", "")   # pre-computed 500-token summary

    # Build full text
    lines = [f"Procedure for: {task_type}"]
    lines.append(f"Trigger: {trigger}")

    if steps:
        for s in steps:
            step_num = s.get("step_num", "?")
            action   = s.get("action", "")
            rationale = s.get("rationale", "")
            tool_hint = s.get("tool_hint", "")
            line = f"  Step {step_num}: {action}"
            if rationale:
                line += f" — {rationale}"
            if tool_hint:
                line += f" [{tool_hint}]"
            lines.append(line)

    if edge_cases:
        lines.append("Edge cases:")
        for ec in edge_cases:
            condition    = ec.get("condition", "")
            modification = ec.get("modification", "")
            lines.append(f"  If {condition}: {modification}")

    if expected:
        lines.append(f"Expected outcome: {expected}")

    full_text = "\n".join(lines)

    if _estimate_tokens(full_text) <= max_tokens:
        return full_text

    # Too long — use pre-computed summary if available
    if summary and _estimate_tokens(summary) <= max_tokens:
        return f"Procedure for: {task_type}\nTrigger: {trigger}\n{summary}"

    # Last resort — truncate at sentence boundaries
    return _truncate_to_sentences(full_text, max_tokens)


# ---------------------------------------------------------------------------
# Semantic memory serializer
# ---------------------------------------------------------------------------
def _serialize_semantic(memory: RawMemory) -> str:
    """
    Serializes a semantic fact into a single annotated line.
    Format: - {fact} [confidence: {conf}]
    """
    props   = memory.properties
    fact    = props.get("fact", memory.content)
    conf    = memory.confidence
    entities = props.get("entities", [])

    line = f"- {fact} [confidence: {conf:.2f}]"
    return line


# ---------------------------------------------------------------------------
# Episodic memory serializer
# ---------------------------------------------------------------------------
def _serialize_episodic(memory: RawMemory) -> str:
    """
    Serializes an episodic memory into a one-line summary.
    Format: - {date}: {task_type} {outcome} — {summary}
    """
    props       = memory.properties
    task_type   = props.get("task_type", "unknown task")
    outcome     = props.get("outcome", "")
    session_start = props.get("session_start", "")
    task_prompt = props.get("task_prompt", memory.content)

    # Short date
    date_str = ""
    if session_start:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(str(session_start).replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except Exception:
            date_str = str(session_start)[:10]

    outcome_str = f" [{outcome.upper()}]" if outcome else ""
    summary = _truncate_to_sentences(task_prompt, 80)   # ~320 chars max

    return f"- {date_str}: {task_type}{outcome_str} — {summary}"


# ---------------------------------------------------------------------------
# Assembly result
# ---------------------------------------------------------------------------
@dataclass
class AssembledContext:
    context_string:    str
    tokens_used:       int
    procedural_found:  bool
    semantic_count:    int
    episodic_count:    int
    memory_ids_used:   list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------
def assemble_context(result: QueryResult) -> AssembledContext:
    """
    Packs reranked memories into prompt text within the token budget.

    Slot reservation rules:
      - Procedural slot (600 tokens) is HARD RESERVE.
        Semantic cannot consume it even if no procedure is found.
        If no procedure, the 600 tokens are redistributed:
          +400 to semantic, +200 to episodic.
      - Never truncate mid-sentence.
      - Total never exceeds BUDGET_HARD_CEILING (2000 tokens).
    """
    sections     = []
    tokens_used  = 0
    ids_used     = []

    procedural_found = result.procedural is not None

    # Resolve slot sizes
    if procedural_found:
        slot_semantic = SLOT_SEMANTIC_BASE
        slot_episodic = SLOT_EPISODIC_BASE
    else:
        slot_semantic = SLOT_SEMANTIC_NO_PROC
        slot_episodic = SLOT_EPISODIC_NO_PROC

    # -----------------------------------------------------------------------
    # SLOT 1 — Procedural (hard reserve 600 tokens)
    # -----------------------------------------------------------------------
    if procedural_found:
        proc_text = _serialize_procedural(result.procedural, SLOT_PROCEDURAL)
        proc_tokens = _estimate_tokens(proc_text)

        sections.append("## Procedure")
        sections.append(proc_text)
        tokens_used += proc_tokens + _estimate_tokens("## Procedure\n")

        pid = result.procedural.postgres_id
        if pid:
            ids_used.append(pid)

    # -----------------------------------------------------------------------
    # SLOT 2 — Semantic facts
    # -----------------------------------------------------------------------
    semantic_lines = []
    semantic_tokens = 0
    semantic_count  = 0

    for mem in result.semantic:
        line = _serialize_semantic(mem)
        cost = _estimate_tokens(line + "\n")

        if semantic_tokens + cost > slot_semantic:
            break

        semantic_lines.append(line)
        semantic_tokens += cost
        semantic_count  += 1

        pid = mem.postgres_id
        if pid:
            ids_used.append(pid)

    if semantic_lines:
        header = "## Known Constraints & Facts\n"
        sections.append("## Known Constraints & Facts")
        sections.extend(semantic_lines)
        tokens_used += semantic_tokens + _estimate_tokens(header)

    # -----------------------------------------------------------------------
    # SLOT 3 — Episodic summaries
    # -----------------------------------------------------------------------
    episodic_lines  = []
    episodic_tokens = 0
    episodic_count  = 0

    for mem in result.episodic:
        line = _serialize_episodic(mem)
        cost = _estimate_tokens(line + "\n")

        if episodic_tokens + cost > slot_episodic:
            break

        episodic_lines.append(line)
        episodic_tokens += cost
        episodic_count  += 1

        pid = mem.postgres_id
        if pid:
            ids_used.append(pid)

    if episodic_lines:
        header = "## Recent Relevant Sessions\n"
        sections.append("## Recent Relevant Sessions")
        sections.extend(episodic_lines)
        tokens_used += episodic_tokens + _estimate_tokens(header)

    # -----------------------------------------------------------------------
    # Final assembly
    # -----------------------------------------------------------------------
    if not sections:
        context_string = ""
    else:
        context_string = "\n".join(sections)

    # Hard ceiling guard — should never trigger with correct slot sizes
    # but defensive truncation just in case
    if _estimate_tokens(context_string) > BUDGET_HARD_CEILING:
        context_string = _truncate_to_sentences(context_string, BUDGET_HARD_CEILING)
        tokens_used = _estimate_tokens(context_string)

    return AssembledContext(
        context_string=context_string,
        tokens_used=tokens_used,
        procedural_found=procedural_found,
        semantic_count=semantic_count,
        episodic_count=episodic_count,
        memory_ids_used=ids_used,
    )