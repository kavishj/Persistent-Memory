"""
core/lifecycle/expiry.py

Soft delete logic for memory lifecycle management.
Implements the two-condition expiry rule from spec Day 3:
  1. Memory must be past its TTL floor (age-based protection)
  2. Importance score must be below expiry threshold

BOTH conditions must be true simultaneously for soft delete.
Exception: hard max age triggers deletion regardless of score.

TTL floors (spec Day 3 — validated against real-world update frequencies):
  Episodic raw:           7 days floor,   90 days hard max
  Semantic constraint:   14 days floor,  180 days hard max
  Semantic preference:   45 days floor,  365 days hard max
  Semantic environment:  14 days floor,  180 days hard max
  Semantic capability:   30 days floor,  365 days hard max
  Semantic relationship: 30 days floor,  365 days hard max
  Semantic global:       60 days floor,  None (manual expiry)
  Procedural active:     90 days floor,  None (manual expiry)
  Procedural degraded:   30 days floor,  365 days hard max
  Procedural failed:      7 days floor,   30 days hard max
  Procedural superseded:  0 days floor,   14 days hard max
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TTL configuration (spec Day 3)
# ---------------------------------------------------------------------------
# (memory_type, subtype) → (floor_days, hard_max_days or None)
TTL_CONFIG: dict[tuple[str, str], tuple[int, Optional[int]]] = {
    # Episodic
    ("episodic",   "raw"):              (7,  90),
    ("episodic",   "archived"):         (30, 365),

    # Semantic by fact_type
    ("semantic",   "constraint"):       (14, 180),
    ("semantic",   "preference"):       (45, 365),
    ("semantic",   "environment"):      (14, 180),
    ("semantic",   "capability"):       (30, 365),
    ("semantic",   "relationship"):     (30, 365),
    ("semantic",   "global"):           (60, None),   # manual expiry only

    # Procedural by state
    ("procedural", "active"):           (90, None),   # manual expiry only
    ("procedural", "degraded"):         (30, 365),
    ("procedural", "failed"):           (7,  30),
    ("procedural", "superseded"):       (0,  14),
}

# Default fallback if subtype not found
TTL_DEFAULT: dict[str, tuple[int, Optional[int]]] = {
    "episodic":   (7,  90),
    "semantic":   (14, 180),
    "procedural": (90, None),
}

# Importance score thresholds for expiry
EXPIRY_THRESHOLD: dict[str, float] = {
    "episodic":   0.25,
    "semantic":   0.20,
    "procedural": 0.30,
}

# Procedural confidence thresholds for state classification
PROC_CONFIDENCE_ACTIVE   = 0.70
PROC_CONFIDENCE_DEGRADED = 0.40


# ---------------------------------------------------------------------------
# Memory subtype resolver
# ---------------------------------------------------------------------------
def _resolve_subtype(
    memory_type:     str,
    fact_type:       Optional[str],
    scope:           Optional[str],
    importance_score: float,
    confidence:      float,
    is_summarized:   bool,
    supersedes:      Optional[str],
) -> str:
    """
    Resolves the TTL subtype for a memory row.
    Used to look up the correct TTL floor and hard max.
    """
    if memory_type == "episodic":
        return "archived" if is_summarized else "raw"

    if memory_type == "semantic":
        if scope == "global":
            return "global"
        return fact_type or "constraint"

    if memory_type == "procedural":
        if supersedes is not None:
            # This memory was superseded by a newer version
            return "superseded"
        if confidence >= PROC_CONFIDENCE_ACTIVE:
            return "active"
        if confidence >= PROC_CONFIDENCE_DEGRADED:
            return "degraded"
        return "failed"

    return "raw"


# ---------------------------------------------------------------------------
# Expiry check result
# ---------------------------------------------------------------------------
@dataclass
class ExpiryDecision:
    should_soft_delete: bool
    reason:             str     # human-readable reason for logging
    age_days:           float
    floor_days:         int
    hard_max_days:      Optional[int]
    importance_score:   float
    threshold:          float


# ---------------------------------------------------------------------------
# Core expiry check
# ---------------------------------------------------------------------------
def should_expire(
    memory_type:      str,
    importance_score: float,
    created_at:       datetime,
    pinned:           bool,
    fact_type:        Optional[str] = None,
    scope:            Optional[str] = None,
    confidence:       float = 0.5,
    is_summarized:    bool = False,
    superseded_by:    Optional[str] = None,
) -> ExpiryDecision:
    """
    Determines whether a memory should be soft-deleted.

    Two-condition rule:
      1. Age >= TTL floor
      2. Importance score < expiry threshold
    Both must be true — except hard max which overrides everything.

    Pinned memories are never expired (manual management only).
    """
    # Pinned — never expire
    if pinned:
        return ExpiryDecision(
            should_soft_delete=False,
            reason="pinned — manual management only",
            age_days=0,
            floor_days=0,
            hard_max_days=None,
            importance_score=importance_score,
            threshold=0,
        )

    # Resolve subtype for TTL lookup
    subtype = _resolve_subtype(
        memory_type, fact_type, scope,
        importance_score, confidence, is_summarized, superseded_by,
    )

    # Lookup TTL config
    key = (memory_type, subtype)
    if key in TTL_CONFIG:
        floor_days, hard_max_days = TTL_CONFIG[key]
    else:
        floor_days, hard_max_days = TTL_DEFAULT.get(
            memory_type, (7, 90)
        )

    # Compute age
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    age_days = (now - created_at).total_seconds() / 86400

    threshold = EXPIRY_THRESHOLD.get(memory_type, 0.25)

    # Hard max — always expire regardless of score
    if hard_max_days is not None and age_days > hard_max_days:
        return ExpiryDecision(
            should_soft_delete=True,
            reason=f"hard max age exceeded ({age_days:.1f}d > {hard_max_days}d)",
            age_days=age_days,
            floor_days=floor_days,
            hard_max_days=hard_max_days,
            importance_score=importance_score,
            threshold=threshold,
        )

    # Below TTL floor — protected
    if age_days < floor_days:
        return ExpiryDecision(
            should_soft_delete=False,
            reason=f"within TTL floor ({age_days:.1f}d < {floor_days}d)",
            age_days=age_days,
            floor_days=floor_days,
            hard_max_days=hard_max_days,
            importance_score=importance_score,
            threshold=threshold,
        )

    # Normal path: both conditions must be true
    below_threshold = importance_score < threshold
    if below_threshold:
        return ExpiryDecision(
            should_soft_delete=True,
            reason=(
                f"importance {importance_score:.4f} < threshold {threshold} "
                f"and past floor ({age_days:.1f}d >= {floor_days}d)"
            ),
            age_days=age_days,
            floor_days=floor_days,
            hard_max_days=hard_max_days,
            importance_score=importance_score,
            threshold=threshold,
        )

    return ExpiryDecision(
        should_soft_delete=False,
        reason=(
            f"importance {importance_score:.4f} >= threshold {threshold} "
            f"(age {age_days:.1f}d, floor {floor_days}d)"
        ),
        age_days=age_days,
        floor_days=floor_days,
        hard_max_days=hard_max_days,
        importance_score=importance_score,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Batch expiry check — used by 24hr lifecycle worker
# ---------------------------------------------------------------------------
def filter_expired(memories: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Splits a list of memory dicts into (to_expire, to_keep).

    Each dict must have:
      - id, memory_type, importance_score, created_at (datetime),
        pinned (bool), fact_type (str|None), scope (str|None),
        confidence (float), ep_is_summarized (bool),
        supersedes (str|None)

    Returns:
      to_expire: list of memory dicts that should be soft-deleted
      to_keep:   list of memory dicts that should be retained
    """
    to_expire = []
    to_keep   = []

    for mem in memories:
        decision = should_expire(
            memory_type=mem.get("memory_type", "episodic"),
            importance_score=float(mem.get("importance_score", 0.5)),
            created_at=mem.get("created_at", datetime.now(timezone.utc)),
            pinned=bool(mem.get("pinned", False)),
            fact_type=mem.get("fact_type"),
            scope=mem.get("scope"),
            confidence=float(mem.get("confidence", 0.5)),
            is_summarized=bool(mem.get("ep_is_summarized", False)),
            superseded_by=mem.get("supersedes"),
        )

        if decision.should_soft_delete:
            logger.debug(
                "Expiry: soft-delete id=%s reason='%s'",
                mem.get("id"), decision.reason,
            )
            to_expire.append(mem)
        else:
            to_keep.append(mem)

    logger.info(
        "Expiry batch: %d to expire, %d to keep (total %d)",
        len(to_expire), len(to_keep), len(memories),
    )
    return to_expire, to_keep