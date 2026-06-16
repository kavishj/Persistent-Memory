"""
Importance score formula — exact implementation from Day 3 design spec.
"""

import math
from datetime import datetime


# Weight tables per memory type (calibrated Day 3)
WEIGHTS = {
    "episodic": {
        "recency": 0.50,
        "access":  0.20,
        "outcome": 0.20,
        "explicit": 0.10,
    },
    "semantic": {
        "recency": 0.25,
        "access":  0.30,
        "outcome": 0.30,
        "explicit": 0.15,
    },
    "procedural": {
        "recency": 0.10,
        "access":  0.35,
        "outcome": 0.40,
        "explicit": 0.15,
    },
}

# Base importance at write time (Day 3)
BASE_IMPORTANCE = {
    ("episodic",   "failure"): 0.72,
    ("episodic",   "partial"): 0.55,
    ("episodic",   "success"): 0.38,
    ("semantic",   "failure"): 0.65,
    ("semantic",   "success"): 0.50,
    ("semantic",   "global"):  0.80,
    ("procedural", "first"):   0.60,
    ("procedural", "update"):  0.75,
    ("procedural", "pinned"):  0.90,
}

# Expiry thresholds (Day 3)
EXPIRY_THRESHOLD = {
    "episodic":   0.25,
    "semantic":   0.20,
    "procedural": 0.30,
}


def recency_score(days_since_confirmed: float, access_count: int) -> float:
    """Ebbinghaus exponential decay with access-frequency stability."""
    stability = min(1.0 + (access_count * 0.5), 10.0)
    return round(math.exp(-days_since_confirmed / stability), 4)


def access_score(access_count: int) -> float:
    """Log-normalized access frequency. Saturates at 50 accesses."""
    return round(min(math.log1p(access_count) / math.log1p(50), 1.0), 4)


def outcome_score(successful_uses: int, total_uses: int) -> float:
    """Success rate. Neutral 0.5 below 3 uses (insufficient evidence)."""
    if total_uses < 3:
        return 0.5
    return round(successful_uses / total_uses, 4)


def compute_importance_score(
    last_confirmed: datetime,
    access_count: int,
    successful_uses: int,
    total_uses: int,
    explicit_signal: float,
    memory_type: str,
    base_importance: float,
) -> float:
    """
    Composite importance score. Range: 0.0–1.0.
    Full formula specification in memory_engine_design_spec.md Day 3.
    """
    now = datetime.utcnow()
    days_since = max((now - last_confirmed).total_seconds() / 86400, 0.0)

    r = recency_score(days_since, access_count)
    a = access_score(access_count)
    o = outcome_score(successful_uses, total_uses)
    e = float(explicit_signal)

    w = WEIGHTS[memory_type]
    weighted = (
        w["recency"]  * r +
        w["access"]   * a +
        w["outcome"]  * o +
        w["explicit"] * e
    )

    # Blend with base importance (write-time prior carries 30%)
    final = (weighted * 0.70) + (base_importance * 0.30)
    return round(min(max(final, 0.0), 1.0), 4)
