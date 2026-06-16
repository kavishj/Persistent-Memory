"""
core/writer/deduplicator.py

Checks extracted facts against existing semantic memories before insert.
Uses cosine similarity to detect duplicates and refinements.

Thresholds (spec Day 3 — validated, DO NOT CHANGE):
  >= 0.92 cosine similarity  → duplicate
  0.75 – 0.92               → refinement (partial overlap)
  < 0.75                    → new fact, insert

Actions:
  duplicate:   update confidence if new is higher, skip insert
  refinement:  insert new + tag existing as 'coarser_version'
  new:         insert normally
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (spec Day 3)
# ---------------------------------------------------------------------------
DUPLICATE_THRESHOLD   = 0.92
REFINEMENT_THRESHOLD  = 0.75


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
class DedupAction(Enum):
    INSERT     = "insert"
    DUPLICATE  = "duplicate"
    REFINEMENT = "refinement"


@dataclass
class DedupResult:
    action:             DedupAction
    existing_id:        Optional[str] = None   # postgres_id of matched memory
    existing_confidence: Optional[float] = None
    new_confidence:     Optional[float] = None
    similarity:         Optional[float] = None
    should_update_confidence: bool = False     # True if duplicate + new conf higher


# ---------------------------------------------------------------------------
# Cosine similarity (pure Python — no numpy dependency)
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Computes cosine similarity between two vectors.
    Returns 0.0 if either vector is empty or zero-magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot    = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a  = math.sqrt(sum(a * a for a in vec_a))
    mag_b  = math.sqrt(sum(b * b for b in vec_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return round(dot / (mag_a * mag_b), 6)


# ---------------------------------------------------------------------------
# Entity overlap signal (spec Day 3 — used alongside cosine similarity)
# ---------------------------------------------------------------------------
def entity_overlap(entities_a: list[str], entities_b: list[str]) -> float:
    """
    Returns Jaccard-style overlap between two entity lists.
    Used as secondary signal to reduce false duplicate detection.
    """
    set_a = {e.lower().strip() for e in entities_a if e}
    set_b = {e.lower().strip() for e in entities_b if e}

    if not set_a and not set_b:
        return 0.5   # neutral when both empty
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union        = len(set_a | set_b)
    return round(intersection / union, 4) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Single fact dedup check
# ---------------------------------------------------------------------------
def check_duplicate(
    new_embedding:     list[float],
    new_confidence:    float,
    new_entities:      list[str],
    existing_memories: list[dict],
) -> DedupResult:
    """
    Checks one extracted fact against a list of existing semantic memories.

    Each item in existing_memories must have:
      - "id":         postgres UUID string
      - "embedding":  list[float] vector
      - "confidence": float
      - "entities":   list[str]

    Returns the DedupResult for the closest match found.
    If no match above REFINEMENT_THRESHOLD, returns INSERT.

    Args:
        new_embedding:     embedding of the new fact
        new_confidence:    confidence of the new fact
        new_entities:      entities in the new fact
        existing_memories: list of existing semantic memory dicts
    """
    if not existing_memories:
        return DedupResult(action=DedupAction.INSERT)

    if not new_embedding:
        # No embedding available — cannot dedup, insert anyway
        logger.warning("Dedup skipped: no embedding for new fact")
        return DedupResult(action=DedupAction.INSERT)

    best_similarity = 0.0
    best_match: Optional[dict] = None

    for mem in existing_memories:
        existing_embedding = mem.get("embedding", [])
        if not existing_embedding:
            continue

        sim = cosine_similarity(new_embedding, existing_embedding)

        if sim > best_similarity:
            best_similarity = sim
            best_match = mem

    if best_match is None or best_similarity < REFINEMENT_THRESHOLD:
        return DedupResult(
            action=DedupAction.INSERT,
            similarity=best_similarity,
        )

    existing_id         = best_match.get("id", "")
    existing_confidence = float(best_match.get("confidence", 0.5))
    existing_entities   = best_match.get("entities", [])

    # Entity overlap as secondary signal
    overlap = entity_overlap(new_entities, existing_entities)

    # ---------------------------------------------------------------------------
    # Decision logic (spec Day 3)
    # ---------------------------------------------------------------------------
    if best_similarity >= DUPLICATE_THRESHOLD:
        # Duplicate — skip insert, optionally update confidence
        should_update = new_confidence > existing_confidence
        logger.debug(
            "Dedup: DUPLICATE (sim=%.4f, entity_overlap=%.4f) existing=%s",
            best_similarity, overlap, existing_id,
        )
        return DedupResult(
            action=DedupAction.DUPLICATE,
            existing_id=existing_id,
            existing_confidence=existing_confidence,
            new_confidence=new_confidence,
            similarity=best_similarity,
            should_update_confidence=should_update,
        )

    else:
        # REFINEMENT_THRESHOLD <= sim < DUPLICATE_THRESHOLD
        # Refinement — insert new, tag existing as coarser
        logger.debug(
            "Dedup: REFINEMENT (sim=%.4f, entity_overlap=%.4f) existing=%s",
            best_similarity, overlap, existing_id,
        )
        return DedupResult(
            action=DedupAction.REFINEMENT,
            existing_id=existing_id,
            existing_confidence=existing_confidence,
            new_confidence=new_confidence,
            similarity=best_similarity,
            should_update_confidence=False,
        )


# ---------------------------------------------------------------------------
# Batch dedup check — used by write path for all facts in one session
# ---------------------------------------------------------------------------
def check_duplicates_batch(
    new_facts: list[dict],
    existing_memories: list[dict],
) -> list[tuple[dict, DedupResult]]:
    """
    Runs dedup check for a batch of new facts.

    Each item in new_facts must have:
      - "embedding":  list[float]
      - "confidence": float
      - "entities":   list[str]
      - "fact":       str (for logging)

    Returns list of (fact, DedupResult) pairs in same order as new_facts.
    """
    results = []
    for fact in new_facts:
        result = check_duplicate(
            new_embedding=fact.get("embedding", []),
            new_confidence=float(fact.get("confidence", 0.5)),
            new_entities=fact.get("entities", []),
            existing_memories=existing_memories,
        )
        results.append((fact, result))
        logger.debug(
            "Dedup: fact='%s...' action=%s",
            str(fact.get("fact", ""))[:40],
            result.action.value,
        )
    return results