"""
core/writer/classifier.py

Classifies:
  1. task_type  — what category of task was this session?
  2. memory_type — should extracted facts be episodic, semantic, or procedural?

Task type classification uses keyword matching + LLM fallback.
Memory type is rule-based (not LLM) — fast and deterministic.

Rules (spec):
  - Episodic:   write after EVERY session regardless of outcome
  - Semantic:   write when extraction confidence >= 0.7
  - Procedural: write only after >= 3 successful sessions of same task_type
                (threshold check is done in write path, not here)
"""

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Known task type keyword map
# Fast path — no LLM call needed for common task types
# ---------------------------------------------------------------------------
TASK_TYPE_KEYWORDS: dict[str, list[str]] = {
    "etl_run":         ["etl", "pipeline", "extract", "transform", "load",
                         "nightly run", "data pipeline"],
    "code_review":     ["review", "pull request", "pr", "diff", "code quality"],
    "data_analysis":   ["analyze", "analysis", "report", "query", "aggregate",
                         "dashboard", "metric"],
    "api_call":        ["api", "endpoint", "request", "response", "http",
                         "rest", "graphql"],
    "file_processing": ["file", "csv", "excel", "parse", "ingest", "upload",
                         "download"],
    "database_query":  ["sql", "query", "select", "insert", "update", "delete",
                         "database", "postgres", "mysql"],
    "deployment":      ["deploy", "release", "publish", "ship", "rollout",
                         "kubernetes", "docker"],
    "monitoring":      ["monitor", "alert", "health check", "status", "uptime",
                         "latency"],
    "data_validation": ["validate", "validation", "schema check", "integrity",
                         "quality check"],
    "summarization":   ["summarize", "summary", "tldr", "condense", "brief"],
}


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------
@dataclass
class ClassificationResult:
    task_type:         str           # e.g. "etl_run", "unknown"
    task_type_source:  str           # "keyword" | "llm" | "default"
    should_write_episodic:  bool     # always True
    should_write_semantic:  bool     # True if any fact confidence >= 0.7
    should_check_procedural: bool    # True if outcome == "success"


# ---------------------------------------------------------------------------
# Task type classifier
# ---------------------------------------------------------------------------
def classify_task_type(task_prompt: str) -> tuple[str, str]:
    """
    Returns (task_type, source) where source is "keyword" | "llm" | "default".

    Keyword match is fast and deterministic — preferred.
    LLM fallback only when no keyword match found.
    """
    prompt_lower = task_prompt.lower()

    # Keyword match
    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in prompt_lower:
                return task_type, "keyword"

    # LLM fallback
    if ANTHROPIC_API_KEY or GEMINI_API_KEY:
        try:
            llm_type = _llm_classify_task_type(task_prompt)
            if llm_type:
                return llm_type, "llm"
        except Exception as e:
            logger.warning("LLM task classification failed: %s", e)

    return "unknown", "default"


def _llm_classify_task_type(task_prompt: str) -> Optional[str]:
    """
    Ask LLM to classify task type.
    Returns a short snake_case label or None on failure.
    """
    system = (
        "You are a task classifier. Given a task prompt, return ONLY a "
        "short snake_case label for the task type (e.g. etl_run, code_review, "
        "data_analysis, api_call, file_processing, database_query, deployment, "
        "monitoring, data_validation, summarization, or a new label if none fit). "
        "Return the label only. No explanation. No punctuation."
    )
    user = f"Task prompt: {task_prompt[:500]}"

    raw = None

    if ANTHROPIC_API_KEY:
        try:
            resp = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 20,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()
        except Exception:
            pass

    if not raw and GEMINI_API_KEY:
        try:
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [{"parts": [{"text": system + "\n\n" + user}]}],
                    "generationConfig": {"maxOutputTokens": 20, "temperature": 0.0},
                },
                timeout=10.0,
            )
            resp.raise_for_status()
            raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception:
            pass

    if not raw:
        return None

    # Sanitize to snake_case
    label = re.sub(r"[^a-z0-9_]", "_", raw.lower()).strip("_")
    return label if label else None


# ---------------------------------------------------------------------------
# Memory type classifier (rule-based — no LLM)
# ---------------------------------------------------------------------------
def classify_memory_writes(
    outcome:            str,
    extracted_facts:    list,        # list[ExtractedFact] — avoid circular import
    min_confidence:     float = 0.7,
) -> ClassificationResult:
    """
    Determines which memory types to write based on spec rules.

    Episodic:   always write (every session)
    Semantic:   write if any extracted fact has confidence >= min_confidence
    Procedural: check if >= 3 successful sessions exist (done in write path)
                Here we just flag: outcome==success means check is warranted

    Args:
        outcome:          "success" | "failure" | "partial"
        extracted_facts:  list of ExtractedFact from extractor.py
        min_confidence:   threshold for semantic write (spec: 0.7)
    """
    should_write_semantic = any(
        f.confidence >= min_confidence for f in extracted_facts
    )
    should_check_procedural = (outcome == "success")

    return ClassificationResult(
        task_type="unknown",              # set by caller after classify_task_type()
        task_type_source="rule",
        should_write_episodic=True,       # always
        should_write_semantic=should_write_semantic,
        should_check_procedural=should_check_procedural,
    )


# ---------------------------------------------------------------------------
# Combined classifier — single call for write path
# ---------------------------------------------------------------------------
def classify_session(
    task_prompt:     str,
    outcome:         str,
    extracted_facts: list,
) -> ClassificationResult:
    """
    Full classification in one call.
    Returns ClassificationResult with all fields populated.
    """
    task_type, source = classify_task_type(task_prompt)
    result = classify_memory_writes(outcome, extracted_facts)
    result.task_type = task_type
    result.task_type_source = source
    return result