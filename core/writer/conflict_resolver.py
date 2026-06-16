"""
core/writer/conflict_resolver.py

Three-case conflict resolution for semantic memories.
Applied AFTER deduplication — only runs on facts that passed dedup
(i.e. similarity 0.60-0.75 range where entity overlap signals conflict).

Cases (spec Day 3 — calibrated values, DO NOT CHANGE):
  Case 1 — Direct contradiction (sim >= 0.92, same entities, different values)
    new confidence >= 0.85 → supersede unconditionally
    new confidence >= 0.70 → supersede + flag for review
    new confidence <  0.70 → reduce existing by 0.15, store new as unresolved

  Case 2 — Partial update (sim 0.75-0.92, matching entities, more specific claim)
    → insert new + tag existing as 'coarser_version'
    → give new fact +0.05 confidence bonus (capped at 1.0)

  Case 3 — Uncertain contradiction (sim 0.60-0.75, overlapping topic)
    → reduce existing confidence by 0.15
    → store new with 'possible_conflict' tag linked to existing

Confidence reduction: 0.15 (NOT 0.20 — spec Day 3 calibration note)
Direct supersede threshold: 0.85
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.writer.deduplicator import cosine_similarity, entity_overlap

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (spec Day 3 — DO NOT CHANGE)
# ---------------------------------------------------------------------------
DIRECT_CONTRADICTION_SIM   = 0.92
PARTIAL_UPDATE_SIM_LOW     = 0.75
UNCERTAIN_SIM_LOW          = 0.60

SUPERSEDE_THRESHOLD        = 0.85   # auto-supersede if new confidence >= this
REVIEW_THRESHOLD           = 0.70   # supersede + flag if new confidence >= this
CONFIDENCE_REDUCTION       = 0.15   # NOT 0.20 — calibrated in Day 3
REFINEMENT_CONFIDENCE_BONUS = 0.05  # bonus for refinement facts


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
class ConflictType(Enum):
    NONE                  = "none"
    DIRECT_CONTRADICTION  = "direct_contradiction"
    PARTIAL_UPDATE        = "partial_update"
    UNCERTAIN             = "uncertain"


class ConflictResolution(Enum):
    NO_CONFLICT           = "no_conflict"
    SUPERSEDE             = "supersede"               # Case 1 high confidence
    SUPERSEDE_FLAG_REVIEW = "supersede_flag_review"   # Case 1 moderate confidence
    WEAKEN_STORE_UNRESOLVED = "weaken_store_unresolved" # Case 1 low confidence
    REFINEMENT            = "refinement"              # Case 2
    POSSIBLE_CONFLICT     = "possible_conflict"       # Case 3


@dataclass
class ConflictResult:
    conflict_type:       ConflictType
    resolution:          ConflictResolution
    existing_id:         Optional[str] = None
    new_confidence:      Optional[float] = None       # adjusted confidence for new fact
    existing_new_confidence: Optional[float] = None  # updated confidence for existing
    conflict_tag:        Optional[str] = None         # tag to set on new memory
    existing_conflict_tag: Optional[str] = None       # tag to set on existing memory
    flag_for_review:     bool = False
    similarity:          float = 0.0
    entity_overlap_score: float = 0.0


# ---------------------------------------------------------------------------
# Conflict type detector (spec Day 3 entity_key method)
# ---------------------------------------------------------------------------
def detect_conflict_type(
    existing_embedding: list[float],
    new_embedding:      list[float],
    existing_entities:  list[str],
    new_entities:       list[str],
) -> tuple[ConflictType, float, float]:
    """
    Returns (ConflictType, similarity, entity_overlap_score).

    Detection uses cosine similarity + entity overlap as combined signal.
    Entity overlap prevents high-similarity topically-related facts
    from being incorrectly flagged as conflicts.
    """
    if not existing_embedding or not new_embedding:
        return ConflictType.NONE, 0.0, 0.0

    sim     = cosine_similarity(existing_embedding, new_embedding)
    overlap = entity_overlap(existing_entities, new_entities)

    # Direct contradiction: very high similarity + entity overlap
    if sim >= DIRECT_CONTRADICTION_SIM and overlap >= 0.5:
        return ConflictType.DIRECT_CONTRADICTION, sim, overlap

    # Partial update: high-ish similarity + moderate entity overlap
    if PARTIAL_UPDATE_SIM_LOW <= sim < DIRECT_CONTRADICTION_SIM and overlap >= 0.3:
        return ConflictType.PARTIAL_UPDATE, sim, overlap

    # Uncertain: moderate similarity + some entity overlap
    if UNCERTAIN_SIM_LOW <= sim < PARTIAL_UPDATE_SIM_LOW and overlap >= 0.2:
        return ConflictType.UNCERTAIN, sim, overlap

    return ConflictType.NONE, sim, overlap


# ---------------------------------------------------------------------------
# Case 1 — Direct contradiction
# ---------------------------------------------------------------------------
def _resolve_direct_contradiction(
    existing_id:         str,
    existing_confidence: float,
    new_confidence:      float,
    similarity:          float,
    entity_overlap_score: float,
) -> ConflictResult:
    """
    New fact directly negates existing fact.

    >= 0.85 new confidence → supersede unconditionally
    >= 0.70 new confidence → supersede + flag for human review
    <  0.70 new confidence → weaken existing by 0.15, store new as unresolved
    """
    if new_confidence >= SUPERSEDE_THRESHOLD:
        logger.info(
            "Conflict Case 1: SUPERSEDE (new_conf=%.2f >= %.2f) existing=%s",
            new_confidence, SUPERSEDE_THRESHOLD, existing_id,
        )
        return ConflictResult(
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            resolution=ConflictResolution.SUPERSEDE,
            existing_id=existing_id,
            new_confidence=new_confidence,
            existing_new_confidence=0.0,       # superseded — score to 0
            conflict_tag="none",               # new fact is clean
            existing_conflict_tag="none",      # will be marked superseded in DB
            flag_for_review=False,
            similarity=similarity,
            entity_overlap_score=entity_overlap_score,
        )

    elif new_confidence >= REVIEW_THRESHOLD:
        logger.info(
            "Conflict Case 1: SUPERSEDE+REVIEW (new_conf=%.2f) existing=%s",
            new_confidence, existing_id,
        )
        return ConflictResult(
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            resolution=ConflictResolution.SUPERSEDE_FLAG_REVIEW,
            existing_id=existing_id,
            new_confidence=new_confidence,
            existing_new_confidence=0.0,
            conflict_tag="review_pending",
            existing_conflict_tag="none",
            flag_for_review=True,
            similarity=similarity,
            entity_overlap_score=entity_overlap_score,
        )

    else:
        # Low confidence — weaken existing, store new as unresolved
        weakened = round(
            max(existing_confidence - CONFIDENCE_REDUCTION, 0.0), 4
        )
        logger.info(
            "Conflict Case 1: WEAKEN+UNRESOLVED (new_conf=%.2f < %.2f) "
            "existing=%s conf %.2f→%.2f",
            new_confidence, REVIEW_THRESHOLD, existing_id,
            existing_confidence, weakened,
        )
        return ConflictResult(
            conflict_type=ConflictType.DIRECT_CONTRADICTION,
            resolution=ConflictResolution.WEAKEN_STORE_UNRESOLVED,
            existing_id=existing_id,
            new_confidence=new_confidence,
            existing_new_confidence=weakened,
            conflict_tag="unresolved_conflict",
            existing_conflict_tag="unresolved_conflict",
            flag_for_review=False,
            similarity=similarity,
            entity_overlap_score=entity_overlap_score,
        )


# ---------------------------------------------------------------------------
# Case 2 — Partial update (refinement)
# ---------------------------------------------------------------------------
def _resolve_partial_update(
    existing_id:          str,
    existing_confidence:  float,
    new_confidence:       float,
    similarity:           float,
    entity_overlap_score: float,
) -> ConflictResult:
    """
    New fact refines existing (more specific claim, same entities).
    Keep both. Tag existing as coarser. Give new a +0.05 confidence bonus.
    """
    adjusted_confidence = round(
        min(new_confidence + REFINEMENT_CONFIDENCE_BONUS, 1.0), 4
    )
    logger.info(
        "Conflict Case 2: REFINEMENT existing=%s "
        "(new_conf %.2f→%.2f with bonus)",
        existing_id, new_confidence, adjusted_confidence,
    )
    return ConflictResult(
        conflict_type=ConflictType.PARTIAL_UPDATE,
        resolution=ConflictResolution.REFINEMENT,
        existing_id=existing_id,
        new_confidence=adjusted_confidence,
        existing_new_confidence=existing_confidence,  # unchanged
        conflict_tag="none",
        existing_conflict_tag="coarser_version",
        flag_for_review=False,
        similarity=similarity,
        entity_overlap_score=entity_overlap_score,
    )


# ---------------------------------------------------------------------------
# Case 3 — Uncertain contradiction
# ---------------------------------------------------------------------------
def _resolve_uncertain(
    existing_id:          str,
    existing_confidence:  float,
    new_confidence:       float,
    similarity:           float,
    entity_overlap_score: float,
) -> ConflictResult:
    """
    Ambiguous overlap — weaken existing, store new with possible_conflict tag.
    Both remain retrievable. Agent sees both and reasons about which applies.
    """
    weakened = round(
        max(existing_confidence - CONFIDENCE_REDUCTION, 0.0), 4
    )
    logger.info(
        "Conflict Case 3: POSSIBLE_CONFLICT existing=%s "
        "conf %.2f→%.2f",
        existing_id, existing_confidence, weakened,
    )
    return ConflictResult(
        conflict_type=ConflictType.UNCERTAIN,
        resolution=ConflictResolution.POSSIBLE_CONFLICT,
        existing_id=existing_id,
        new_confidence=new_confidence,
        existing_new_confidence=weakened,
        conflict_tag="possible_conflict",
        existing_conflict_tag="possible_conflict",
        flag_for_review=False,
        similarity=similarity,
        entity_overlap_score=entity_overlap_score,
    )


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------
def resolve_conflict(
    existing_id:         str,
    existing_embedding:  list[float],
    existing_confidence: float,
    existing_entities:   list[str],
    new_embedding:       list[float],
    new_confidence:      float,
    new_entities:        list[str],
) -> ConflictResult:
    """
    Full conflict detection + resolution in one call.

    Returns ConflictResult describing what action the write path should take:
      - Update existing memory (confidence, conflict_tag, supersedes)
      - Set tag on new memory before insert
      - Flag for human review

    If no conflict detected, returns NO_CONFLICT result.
    """
    conflict_type, sim, overlap = detect_conflict_type(
        existing_embedding, new_embedding,
        existing_entities, new_entities,
    )

    if conflict_type == ConflictType.NONE:
        return ConflictResult(
            conflict_type=ConflictType.NONE,
            resolution=ConflictResolution.NO_CONFLICT,
            similarity=sim,
            entity_overlap_score=overlap,
        )

    if conflict_type == ConflictType.DIRECT_CONTRADICTION:
        return _resolve_direct_contradiction(
            existing_id, existing_confidence,
            new_confidence, sim, overlap,
        )

    if conflict_type == ConflictType.PARTIAL_UPDATE:
        return _resolve_partial_update(
            existing_id, existing_confidence,
            new_confidence, sim, overlap,
        )

    # ConflictType.UNCERTAIN
    return _resolve_uncertain(
        existing_id, existing_confidence,
        new_confidence, sim, overlap,
    )


# ---------------------------------------------------------------------------
# Batch resolver — check one new fact against all existing memories
# ---------------------------------------------------------------------------
def resolve_conflicts_batch(
    new_embedding:   list[float],
    new_confidence:  float,
    new_entities:    list[str],
    existing_memories: list[dict],
) -> list[ConflictResult]:
    """
    Checks one new fact against all existing memories.
    Returns only non-trivial conflict results (excludes NO_CONFLICT).

    Each item in existing_memories must have:
      - "id":         str
      - "embedding":  list[float]
      - "confidence": float
      - "entities":   list[str]
    """
    conflicts = []
    for mem in existing_memories:
        result = resolve_conflict(
            existing_id=mem.get("id", ""),
            existing_embedding=mem.get("embedding", []),
            existing_confidence=float(mem.get("confidence", 0.5)),
            existing_entities=mem.get("entities", []),
            new_embedding=new_embedding,
            new_confidence=new_confidence,
            new_entities=new_entities,
        )
        if result.resolution != ConflictResolution.NO_CONFLICT:
            conflicts.append(result)

    return conflicts