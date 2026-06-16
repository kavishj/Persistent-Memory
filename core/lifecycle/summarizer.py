"""
core/lifecycle/summarizer.py
 
Episodic → Semantic promotion logic.
Spec Day 3: summarization pipeline.
 
Trigger conditions (ALL must be true):
  1. episodic memory is NOT already summarized (ep_is_summarized=False)
  2. age >= SUMMARIZATION_AGE_FLOOR (3 days)
  3. importance_score >= SUMMARIZATION_IMPORTANCE_MIN (0.40)
  4. memory is not soft-deleted
 
Promotion flow:
  1. Pull eligible episodic rows from Postgres
  2. Group by agent_id + task_type (batch up to BATCH_SIZE per group)
  3. Call Anthropic (claude-haiku-4-5) with SUMMARIZATION_SYSTEM_PROMPT
     — Gemini fallback on failure (fail-open: skip if both fail)
  4. Parse response → list of semantic fact dicts
  5. Write semantic memories to Postgres + queue Weaviate embed
  6. Mark episodic rows ep_is_summarized=True, ep_summarized_at=now
 
Constants (spec Day 3):
  SUMMARIZATION_AGE_FLOOR          = 3 days
  SUMMARIZATION_IMPORTANCE_MIN     = 0.40
  BATCH_SIZE                       = 10 episodic memories per LLM call
  MAX_SEMANTIC_FACTS_PER_BATCH     = 5
  SUMMARIZATION_MODEL_PRIMARY      = claude-haiku-4-5
  SUMMARIZATION_MODEL_FALLBACK     = gemini-2.0-flash
"""
 
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
 
try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None
try:
    from google import genai
except ModuleNotFoundError:
    genai = None
 
logger = logging.getLogger(__name__)
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUMMARIZATION_AGE_FLOOR          = 3       # days
SUMMARIZATION_IMPORTANCE_MIN     = 0.40
BATCH_SIZE                       = 10
MAX_SEMANTIC_FACTS_PER_BATCH     = 5
SUMMARIZATION_MODEL_PRIMARY      = "claude-haiku-4-5"
SUMMARIZATION_MODEL_FALLBACK     = "gemini-2.0-flash"
 
# ---------------------------------------------------------------------------
# Prompt (matches extractor.py SUMMARIZATION_SYSTEM_PROMPT — spec Day 2)
# ---------------------------------------------------------------------------
SUMMARIZATION_SYSTEM_PROMPT = """You are a memory distillation engine. Your job is to extract durable semantic facts from episodic memory records.
 
Given a batch of episodic memories from a single agent working on a specific task type, extract up to {max_facts} semantic facts that are:
- Stable and reusable across future sessions (not one-off observations)
- Factual claims about capabilities, constraints, preferences, environment, or relationships
- Expressed as concise, standalone statements
 
For each fact output:
- content: the fact as a clear declarative sentence
- fact_type: one of [constraint, preference, environment, capability, relationship]
- confidence: 0.0–1.0 (how certain this fact is based on evidence)
- supporting_memory_ids: list of episodic memory IDs that support this fact
 
Respond ONLY with a JSON array. No preamble, no markdown fences. Example:
[
  {
    "content": "Agent cannot access external URLs during task execution.",
    "fact_type": "constraint",
    "confidence": 0.90,
    "supporting_memory_ids": ["uuid-1", "uuid-2"]
  }
]"""
 
SUMMARIZATION_USER_TEMPLATE = """Agent: {agent_id}
Task type: {task_type}
Memory batch ({count} memories):
 
{memories_json}
 
Extract up to {max_facts} durable semantic facts."""
 
 
# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SemanticFact:
    content:               str
    fact_type:             str        # constraint|preference|environment|capability|relationship
    confidence:            float
    supporting_memory_ids: list[str]
    agent_id:              str
    task_type_id:          Optional[str]
 
 
@dataclass
class SummarizationResult:
    agent_id:              str
    task_type:             str
    episodic_ids_processed: list[str]
    semantic_facts_created: list[SemanticFact]
    failed:                bool = False
    failure_reason:        str = ""
 
 
# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------
def _call_primary(prompt_system: str, prompt_user: str) -> str:
    """claude-haiku-4-5 call. Returns raw text or raises."""
    if anthropic is None:
        raise ModuleNotFoundError(
            "anthropic is not installed; primary summarization model is unavailable"
        )
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = client.messages.create(
        model=SUMMARIZATION_MODEL_PRIMARY,
        max_tokens=1024,
        system=prompt_system,
        messages=[{"role": "user", "content": prompt_user}],
    )
    return msg.content[0].text
 
 
def _call_fallback(prompt_system: str, prompt_user: str) -> str:
    """gemini-2.0-flash fallback. Returns raw text or raises."""
    if genai is None:
        raise ModuleNotFoundError(
            "google.genai is not installed; Gemini fallback is unavailable"
        )
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(
        model=SUMMARIZATION_MODEL_FALLBACK,
        contents=prompt_user,
        config={
            "system_instruction": prompt_system,
        },
    )
    return resp.text
 
 
def _call_llm_with_fallback(prompt_system: str, prompt_user: str) -> Optional[str]:
    """
    Try primary (Anthropic), fallback to Gemini.
    Fail-open: return None if both fail.
    """
    try:
        return _call_primary(prompt_system, prompt_user)
    except Exception as e:
        logger.warning("Summarizer primary LLM failed: %s — trying fallback", e)
 
    try:
        return _call_fallback(prompt_system, prompt_user)
    except Exception as e:
        logger.error("Summarizer fallback LLM failed: %s — skipping batch", e)
        return None
 
 
# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
def _parse_facts(
    raw:          str,
    agent_id:     str,
    task_type_id: Optional[str],
    max_facts:    int,
) -> list[SemanticFact]:
    """
    Parse LLM JSON response → list[SemanticFact].
    Validates required fields. Clamps confidence 0–1.
    Returns [] on any parse error (fail-open).
    """
    try:
        # Strip accidental markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )
        data = json.loads(clean)
        if not isinstance(data, list):
            raise ValueError("Expected JSON array")
 
        facts = []
        for item in data[:max_facts]:
            if not isinstance(item, dict):
                continue
            content   = str(item.get("content", "")).strip()
            fact_type = str(item.get("fact_type", "constraint")).strip()
            confidence = float(item.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            supporting = [
                str(x) for x in item.get("supporting_memory_ids", [])
            ]
 
            if not content:
                continue
            if fact_type not in (
                "constraint", "preference", "environment",
                "capability", "relationship"
            ):
                fact_type = "constraint"
 
            facts.append(SemanticFact(
                content=content,
                fact_type=fact_type,
                confidence=confidence,
                supporting_memory_ids=supporting,
                agent_id=agent_id,
                task_type_id=task_type_id,
            ))
        return facts
 
    except Exception as e:
        logger.error("Summarizer parse error: %s | raw=%r", e, raw[:200])
        return []
 
 
# ---------------------------------------------------------------------------
# Eligibility filter
# ---------------------------------------------------------------------------
def filter_eligible(memories: list[dict]) -> list[dict]:
    """
    Filter episodic memories eligible for summarization.
    Conditions: not summarized, not soft-deleted, age >= floor, score >= min.
    """
    now = datetime.now(timezone.utc)
    eligible = []
    for m in memories:
        if m.get("ep_is_summarized"):
            continue
        if m.get("deleted_at") is not None:
            continue
        created_at = m.get("created_at", now)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        age_days = (now - created_at).total_seconds() / 86400
        if age_days < SUMMARIZATION_AGE_FLOOR:
            continue
        if float(m.get("importance_score", 0)) < SUMMARIZATION_IMPORTANCE_MIN:
            continue
        eligible.append(m)
    return eligible
 
 
# ---------------------------------------------------------------------------
# Single-batch summarizer
# ---------------------------------------------------------------------------
def summarize_batch(
    agent_id:     str,
    task_type:    str,
    task_type_id: Optional[str],
    memories:     list[dict],
) -> SummarizationResult:
    """
    Summarize up to BATCH_SIZE episodic memories → semantic facts.
    memories: list of memory dicts with at least {id, content, created_at,
              importance_score, outcome_feedback}.
    """
    batch = memories[:BATCH_SIZE]
    episodic_ids = [str(m["id"]) for m in batch]
 
    # Build memories JSON for prompt (minimal fields)
    memories_for_prompt = [
        {
            "id":      str(m["id"]),
            "content": m.get("content", ""),
            "outcome": m.get("outcome_feedback", "unknown"),
        }
        for m in batch
    ]
 
    prompt_system = SUMMARIZATION_SYSTEM_PROMPT.format(
        max_facts=MAX_SEMANTIC_FACTS_PER_BATCH
    )
    prompt_user = SUMMARIZATION_USER_TEMPLATE.format(
        agent_id=agent_id,
        task_type=task_type,
        count=len(batch),
        memories_json=json.dumps(memories_for_prompt, indent=2),
        max_facts=MAX_SEMANTIC_FACTS_PER_BATCH,
    )
 
    raw = _call_llm_with_fallback(prompt_system, prompt_user)
    if raw is None:
        return SummarizationResult(
            agent_id=agent_id,
            task_type=task_type,
            episodic_ids_processed=episodic_ids,
            semantic_facts_created=[],
            failed=True,
            failure_reason="both LLMs failed",
        )
 
    facts = _parse_facts(raw, agent_id, task_type_id, MAX_SEMANTIC_FACTS_PER_BATCH)
 
    logger.info(
        "Summarizer: agent=%s task_type=%s episodic=%d → semantic_facts=%d",
        agent_id, task_type, len(batch), len(facts),
    )
 
    return SummarizationResult(
        agent_id=agent_id,
        task_type=task_type,
        episodic_ids_processed=episodic_ids,
        semantic_facts_created=facts,
    )
 
 
# ---------------------------------------------------------------------------
# Multi-batch driver — called by Celery worker
# ---------------------------------------------------------------------------
def process_summarization_queue(memories_by_group: dict[tuple[str, str, Optional[str]], list[dict]]) -> list[SummarizationResult]:
    """
    Process all pending summarization groups.
 
    memories_by_group: {(agent_id, task_type, task_type_id): [memory_dicts]}
    Each group processed in BATCH_SIZE chunks.
    Returns list of SummarizationResults (one per batch chunk).
 
    Caller (workers/tasks.py) is responsible for:
      - Querying eligible episodic memories from Postgres
      - Grouping by (agent_id, task_type)
      - Writing returned SemanticFacts to Postgres
      - Marking episodic rows ep_is_summarized=True
      - Queuing Weaviate embed jobs for new semantic memories
    """
    results = []
 
    for (agent_id, task_type, task_type_id), memories in memories_by_group.items():
        eligible = filter_eligible(memories)
        if not eligible:
            continue
 
        # Chunk into BATCH_SIZE groups
        for i in range(0, len(eligible), BATCH_SIZE):
            chunk = eligible[i : i + BATCH_SIZE]
            result = summarize_batch(agent_id, task_type, task_type_id, chunk)
            results.append(result)
 
    return results
