"""
core/retrieval/reranker.py

Reranks raw Weaviate results using the three-signal formula:
  final_rank = retrieval_score × 0.5
             + importance_score × 0.3
             + recency_score × 0.2

Recency score reuses the same Ebbinghaus formula from the lifecycle scorer
to stay consistent — a memory's recency means the same thing everywhere.

Spec constants (DO NOT CHANGE):
  retrieval weight:  0.5
  importance weight: 0.3
  recency weight:    0.2
  k after rerank:    semantic=5, episodic=3
"""

import math
from datetime import datetime, timezone
from typing import Optional

from core.retrieval.query_builder import QueryResult, RawMemory, K_SEMANTIC, K_EPISODIC


# ---------------------------------------------------------------------------
# Rerank weights (spec Day 6)
# ---------------------------------------------------------------------------
W_RETRIEVAL  = 0.5
W_IMPORTANCE = 0.3
W_RECENCY    = 0.2


# ---------------------------------------------------------------------------
# Recency score (same formula as lifecycle scorer — must stay in sync)
# ---------------------------------------------------------------------------
def _recency_score(last_confirmed: Optional[datetime], access_count: int) -> float:
    """
    Ebbinghaus decay with access-frequency stability modifier.
    Matches compute_importance_score() in core/lifecycle/scorer.py exactly.

    If last_confirmed is unknown, defaults to 0.5 (neutral).
    """
    if last_confirmed is None:
        return 0.5

    now = datetime.now(timezone.utc)

    # Make last_confirmed timezone-aware if naive
    if last_confirmed.tzinfo is None:
        last_confirmed = last_confirmed.replace(tzinfo=timezone.utc)

    days = max((now - last_confirmed).total_seconds() / 86400, 0.0)
    stability = min(1.0 + (access_count * 0.5), 10.0)
    return round(math.exp(-days / stability), 4)


# ---------------------------------------------------------------------------
# Single memory rerank score
# ---------------------------------------------------------------------------
def _final_score(memory: RawMemory) -> float:
    """
    Compute final rerank score for one memory.
    All three signals normalized to [0, 1].
    """
    access_count = int(memory.properties.get("access_count", 0))
    last_confirmed_raw = memory.properties.get("last_confirmed")

    if isinstance(last_confirmed_raw, str):
        try:
            last_confirmed = datetime.fromisoformat(
                last_confirmed_raw.replace("Z", "+00:00")
            )
        except ValueError:
            last_confirmed = None
    elif isinstance(last_confirmed_raw, datetime):
        last_confirmed = last_confirmed_raw
    else:
        last_confirmed = None

    r = min(max(memory.retrieval_score, 0.0), 1.0)
    i = min(max(memory.importance_score, 0.0), 1.0)
    c = _recency_score(last_confirmed, access_count)

    score = (W_RETRIEVAL * r) + (W_IMPORTANCE * i) + (W_RECENCY * c)
    return round(min(max(score, 0.0), 1.0), 4)


# ---------------------------------------------------------------------------
# Deduplication by postgres_id
# ---------------------------------------------------------------------------
def _deduplicate(memories: list[RawMemory]) -> list[RawMemory]:
    """
    Remove duplicate postgres_ids — can occur when both agent + global
    tenants return the same promoted fact.
    Keeps the higher-scoring copy.
    """
    seen: dict[str, RawMemory] = {}
    for m in memories:
        pid = m.postgres_id
        if not pid:
            continue
        if pid not in seen or _final_score(m) > _final_score(seen[pid]):
            seen[pid] = m
    return list(seen.values())


# ---------------------------------------------------------------------------
# Main reranker
# ---------------------------------------------------------------------------
def rerank(result: QueryResult) -> QueryResult:
    """
    Takes raw QueryResult from query_builder and returns a new QueryResult
    with memories sorted by final_rank and trimmed to spec k values.

    Procedural is not reranked — it is already the single best match
    (or None) from query_builder. Confidence is its primary signal.

    Returns the same QueryResult structure so context_assembler
    receives a consistent interface.
    """

    # --- Semantic ---
    deduped_semantic = _deduplicate(result.semantic)
    ranked_semantic = sorted(
        deduped_semantic,
        key=_final_score,
        reverse=True,
    )
    top_semantic = ranked_semantic[:K_SEMANTIC]

    # --- Episodic ---
    # No dedup needed — episodes are unique per session
    ranked_episodic = sorted(
        result.episodic,
        key=_final_score,
        reverse=True,
    )
    top_episodic = ranked_episodic[:K_EPISODIC]

    return QueryResult(
        semantic=top_semantic,
        procedural=result.procedural,   # pass through unchanged
        episodic=top_episodic,
        embed_ms=result.embed_ms,
        query_ms=result.query_ms,
        cache_hit=result.cache_hit,
    )


# ---------------------------------------------------------------------------
# Debug helper — show scores for a result set
# ---------------------------------------------------------------------------
def explain_ranking(result: QueryResult) -> str:
    """
    Returns a human-readable breakdown of rerank scores.
    Useful during Day 11 benchmarking.
    """
    lines = ["=== Rerank Explanation ===\n"]

    lines.append("SEMANTIC:")
    for i, m in enumerate(result.semantic):
        score = _final_score(m)
        r = min(max(m.retrieval_score, 0.0), 1.0)
        imp = m.importance_score
        rec = _recency_score(None, 0)
        lines.append(
            f"  [{i+1}] score={score:.4f} "
            f"(ret={r:.3f}×0.5 + imp={imp:.3f}×0.3 + rec={rec:.3f}×0.2) "
            f"| {m.content[:60]}"
        )

    lines.append("\nPROCEDURAL:")
    if result.procedural:
        p = result.procedural
        lines.append(
            f"  [1] conf={p.confidence:.3f} "
            f"imp={p.importance_score:.3f} "
            f"| {p.content[:60]}"
        )
    else:
        lines.append("  None found above threshold")

    lines.append("\nEPISODIC:")
    for i, m in enumerate(result.episodic):
        score = _final_score(m)
        lines.append(
            f"  [{i+1}] score={score:.4f} "
            f"imp={m.importance_score:.3f} "
            f"| {m.content[:60]}"
        )

    return "\n".join(lines)