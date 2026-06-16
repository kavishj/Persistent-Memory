"""
core/writer/extractor.py

Extracts semantic facts from completed agent session logs using an LLM.
Uses the exact SUMMARIZATION_SYSTEM_PROMPT from the spec (Day 3).

LLM priority:
  1. Groq llama-3.1-8b-instant   (primary)
  2. Anthropic claude-haiku-4-5  (fallback)
  3. Gemini gemini-2.0-flash     (second fallback)

Extraction rules (spec):
  - Max 10 facts per session
  - Min 0 facts (return [] if nothing worth storing)
  - Confidence floor: 0.6
  - Output: JSON array only — no preamble, no markdown
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL       = "https://api.groq.com/openai/v1/chat/completions"

CONFIDENCE_FLOOR = 0.6
MAX_FACTS        = 10
EXTRACT_TIMEOUT  = 30.0

VALID_FACT_TYPES = {
    "constraint", "preference", "environment",
    "capability", "relationship"
}

# ---------------------------------------------------------------------------
# Exact summarization prompt from spec (Day 3) — DO NOT PARAPHRASE
# ---------------------------------------------------------------------------
SUMMARIZATION_SYSTEM_PROMPT = """You are a memory extraction system for an AI agent.
Output ONLY a JSON array. No preamble. No explanation.
No markdown code fences. Raw JSON only.

WHAT TO PRESERVE:
- Constraints discovered: rate limits, size limits, timeouts, quotas
- Tool behaviors: what worked, what failed, under what conditions
- Environment facts: service versions, config values, dependencies
- User or system preferences explicitly stated or strongly implied
- Error patterns: what caused failures and what resolved them
- Workflow patterns: if a multi-step sequence succeeded, distill the key steps
  and the conditions that made it work

WHAT TO DISCARD:
- Intermediate reasoning steps that led to dead ends
- Exact parameter values unless they reveal a constraint
- Verbose tool outputs -- extract the meaning, not the text
- Routine successful steps with no learning signal
- Anything that would be obvious from the task type alone

FACT QUALITY RULES:
- Each fact must be atomic: one claim per fact object
- Facts must be stated in plain language, not jargon
- Facts must be falsifiable -- not opinions or preferences about style
- Confidence reflects how certain the session evidence makes you
  (0.9+ = directly stated; 0.7-0.9 = strongly implied; 0.6-0.7 = inferred)
- Do not include facts with confidence below 0.6

OUTPUT SCHEMA (return this exact structure):
[
  {
    "fact": "one atomic fact in plain language",
    "fact_type": "constraint|preference|environment|capability|relationship",
    "entities": ["named entities mentioned in this fact"],
    "confidence": 0.0-1.0,
    "source": "brief phrase: which part of the session produced this fact"
  }
]

LIMITS:
- Maximum 10 facts per session
- Minimum 0 facts (return [] if nothing is worth storing)
- If in doubt about a fact's value: omit it"""

SUMMARIZATION_USER_TEMPLATE = """SESSION METADATA:
agent_id: {agent_id}
task_type: {task_type}
outcome: {outcome}
duration_ms: {duration_ms}
session_start: {session_start}

SESSION LOG:
{session_log}

Extract facts worth retaining in long-term memory."""


# ---------------------------------------------------------------------------
# Extracted fact dataclass
# ---------------------------------------------------------------------------
@dataclass
class ExtractedFact:
    fact:       str
    fact_type:  str
    entities:   list[str]
    confidence: float
    source:     str

    def is_valid(self) -> bool:
        return (
            bool(self.fact.strip())
            and self.fact_type in VALID_FACT_TYPES
            and self.confidence >= CONFIDENCE_FLOOR
        )


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------
def _call_anthropic(user_prompt: str) -> str:
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": ANTHROPIC_MODEL,
            "max_tokens": 1024,
            "system": SUMMARIZATION_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=EXTRACT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_gemini(user_prompt: str) -> str:
    full_prompt = SUMMARIZATION_SYSTEM_PROMPT + "\n\n" + user_prompt
    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}",
        json={
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.1},
        },
        timeout=EXTRACT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(user_prompt: str) -> str:
    """Groq — OpenAI-compatible API, llama-3.1-8b-instant (free tier)."""
    resp = httpx.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SUMMARIZATION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens":  1024,
            "temperature": 0.1,
        },
        timeout=EXTRACT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------
def _parse_response(raw: str) -> list[ExtractedFact]:
    text = raw.strip()
    
    # robust json extraction
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    else:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and start < end:
            text = text[start:end+1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Extraction parse failed: %s | raw=%s", e, text[:200])
        return []

    if not isinstance(data, list):
        logger.warning("Extraction returned non-list: %s", type(data))
        return []

    facts = []
    for item in data[:MAX_FACTS]:
        if not isinstance(item, dict):
            continue
        fact = ExtractedFact(
            fact=str(item.get("fact", "")).strip(),
            fact_type=str(item.get("fact_type", "constraint")).lower().strip(),
            entities=[str(e) for e in item.get("entities", [])],
            confidence=float(item.get("confidence", 0.0)),
            source=str(item.get("source", "")).strip(),
        )
        if fact.is_valid():
            facts.append(fact)
        else:
            logger.debug(
                "Filtered invalid fact: %s (conf=%.2f type=%s)",
                fact.fact[:50], fact.confidence, fact.fact_type,
            )
    return facts


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------
def extract_facts(
    session_log:   str,
    agent_id:      str,
    task_type:     str = "unknown",
    outcome:       str = "success",
    duration_ms:   Optional[int] = None,
    session_start: Optional[str] = None,
) -> list[ExtractedFact]:
    """
    Extracts semantic facts from a session log.
    Priority: Groq → Anthropic → Gemini.
    Returns [] if all fail — write path stores episodic only.
    """
    user_prompt = SUMMARIZATION_USER_TEMPLATE.format(
        agent_id=agent_id,
        task_type=task_type,
        outcome=outcome,
        duration_ms=duration_ms or "unknown",
        session_start=session_start or "unknown",
        session_log=session_log,
    )

    # 1. Groq
    if GROQ_API_KEY:
        try:
            raw   = _call_groq(user_prompt)
            facts = _parse_response(raw)
            logger.info(
                "Extraction (groq): %d facts [agent=%s task=%s]",
                len(facts), agent_id, task_type,
            )
            return facts
        except Exception as e:
            logger.warning("Groq extraction failed: %s — trying Anthropic", e)

    # 2. Anthropic
    if ANTHROPIC_API_KEY:
        try:
            raw   = _call_anthropic(user_prompt)
            facts = _parse_response(raw)
            logger.info(
                "Extraction (anthropic): %d facts [agent=%s task=%s]",
                len(facts), agent_id, task_type,
            )
            return facts
        except Exception as e:
            logger.warning("Anthropic extraction failed: %s — trying Gemini", e)

    # 3. Gemini
    if GEMINI_API_KEY:
        try:
            raw   = _call_gemini(user_prompt)
            facts = _parse_response(raw)
            logger.info(
                "Extraction (gemini): %d facts [agent=%s task=%s]",
                len(facts), agent_id, task_type,
            )
            return facts
        except Exception as e:
            logger.warning("Gemini extraction failed: %s", e)

    logger.error(
        "All extraction backends failed [agent=%s task=%s]. Storing episodic only.",
        agent_id, task_type,
    )
    return []


# ---------------------------------------------------------------------------
# Strict retry
# ---------------------------------------------------------------------------
def extract_facts_strict(
    session_log: str,
    agent_id:    str,
    **kwargs,
) -> list[ExtractedFact]:
    strict_log = (
        "IMPORTANT: Return ONLY a raw JSON array. "
        "No text before or after. No markdown. Start with [ end with ].\n\n"
        + session_log
    )
    return extract_facts(strict_log, agent_id, **kwargs)
