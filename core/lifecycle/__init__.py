"""
core/lifecycle/__init__.py

Lifecycle module: scoring, expiry, summarization, reconciliation.
Spec Day 3.
"""

from core.lifecycle.scorer import compute_importance_score
score_memory = compute_importance_score  # alias
from core.lifecycle.expiry import (
    should_expire,
    filter_expired,
    ExpiryDecision,
    TTL_CONFIG,
    EXPIRY_THRESHOLD,
)
from core.lifecycle.summarizer import (
    process_summarization_queue,
    summarize_batch,
    filter_eligible,
    SummarizationResult,
    SemanticFact,
    SUMMARIZATION_AGE_FLOOR,
    SUMMARIZATION_IMPORTANCE_MIN,
)
from core.lifecycle.reconciler import (
    reconcile_batch,
    decide_repair,
    decide_orphan,
    find_orphaned_weaviate_objects,
    ReconcileDecision,
    ReconciliationReport,
    RepairAction,
)

__all__ = [
    # scorer
    "compute_importance_score",
    "score_memory",
    # expiry
    "should_expire", "filter_expired", "ExpiryDecision",
    "TTL_CONFIG", "EXPIRY_THRESHOLD",
    # summarizer
    "process_summarization_queue", "summarize_batch", "filter_eligible",
    "SummarizationResult", "SemanticFact",
    "SUMMARIZATION_AGE_FLOOR", "SUMMARIZATION_IMPORTANCE_MIN",
    # reconciler
    "reconcile_batch", "decide_repair", "decide_orphan",
    "find_orphaned_weaviate_objects",
    "ReconcileDecision", "ReconciliationReport", "RepairAction",
]
 