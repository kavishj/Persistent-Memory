"""
core/retrieval/query_builder.py

Embeds the task prompt and builds parallel Weaviate hybrid queries
for all three memory types.

Spec constants (DO NOT CHANGE without re-benchmarking):
  alpha:  semantic=0.7, procedural=0.4, episodic=0.5
  k:      semantic=5,   procedural=1,   episodic=3
  HNSW ef already set at schema creation — not repeated here.

Embedding: sentence-transformers/all-MiniLM-L6-v2 (384 dims, local, free)
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import redis


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEAVIATE_URL     = os.getenv("WEAVIATE_URL", "http://localhost:8080")
REDIS_URL        = os.getenv("REDIS_URL", "redis://localhost:6379/0")
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIMS       = 384
EMBED_CACHE_TTL  = 3600         # 1 hour Redis TTL on embeddings

# Validated alpha values (Day 2 benchmark)
ALPHA_SEMANTIC   = 0.7
ALPHA_PROCEDURAL = 0.4
ALPHA_EPISODIC   = 0.5

# Validated k values (Day 2 benchmark)
K_SEMANTIC       = 5
K_PROCEDURAL     = 1
K_EPISODIC       = 3
K_SEMANTIC_OVER  = 10

# Retrieval thresholds
MIN_CONFIDENCE_SEMANTIC   = 0.50
MIN_CONFIDENCE_PROCEDURAL = 0.60
EPISODIC_DAYS_BACK        = 30

# Lazy-loaded model singleton
_model = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RawMemory:
    """Single memory object returned from Weaviate before reranking."""
    postgres_id:      str
    memory_type:      str
    content:          str
    retrieval_score:  float
    properties:       dict = field(default_factory=dict)
    importance_score: float = 0.5
    confidence:       float = 0.5
    last_confirmed:   Optional[datetime] = None


@dataclass
class QueryResult:
    """Output of build_queries() — raw results for all three types."""
    semantic:    list[RawMemory] = field(default_factory=list)
    procedural:  Optional[RawMemory] = None
    episodic:    list[RawMemory] = field(default_factory=list)
    embed_ms:    int = 0
    query_ms:    int = 0
    cache_hit:   bool = False


# ---------------------------------------------------------------------------
# Embedding — sentence-transformers (local, free, 384 dims)
# ---------------------------------------------------------------------------
def _get_model():
    """Lazy-load model singleton — loads once, reused across calls."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _redis_client() -> redis.Redis:
    return redis.from_url(REDIS_URL, decode_responses=False)


def _cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return f"embed384:{digest}"


def embed_text(text: str) -> tuple[list[float], bool]:
    """
    Returns (embedding_vector, cache_hit).
    Checks Redis first. Falls back to local sentence-transformers model.
    Returns empty list on error (caller handles fail-open).
    """
    key = _cache_key(text)

    # Cache check
    try:
        r = _redis_client()
        cached = r.get(key)
        if cached:
            vector = json.loads(cached)
            return vector, True
    except Exception:
        pass

    # Local model inference
    try:
        model  = _get_model()
        vector = model.encode(text, normalize_embeddings=True).tolist()

        # Cache result
        try:
            r = _redis_client()
            r.setex(key, EMBED_CACHE_TTL, json.dumps(vector))
        except Exception:
            pass

        return vector, False

    except Exception as e:
        import logging
        logging.getLogger(__name__).error("embed_text failed: %s", e)
        return [], False


# ---------------------------------------------------------------------------
# Weaviate query helpers
# ---------------------------------------------------------------------------
def _get_weaviate_collection(name: str, tenant: str):
    import weaviate
    client = weaviate.connect_to_local(host="localhost", port=8080)
    collection = client.collections.get(name)
    return client, collection.with_tenant(tenant)


def _query_semantic(
    agent_id: str,
    query_text: str,
    query_vector: list[float],
) -> list[RawMemory]:
    from weaviate.classes.query import MetadataQuery, HybridFusion, Filter

    results: list[RawMemory] = []

    for tenant in [agent_id, "__global__"]:
        try:
            client, col = _get_weaviate_collection("SemanticMemory", tenant)
            try:
                resp = col.query.hybrid(
                    query=query_text,
                    vector=query_vector if query_vector else None,
                    alpha=ALPHA_SEMANTIC,
                    limit=K_SEMANTIC_OVER,
                    fusion_type=HybridFusion.RELATIVE_SCORE,
                    return_metadata=MetadataQuery(score=True),
                    filters=Filter.by_property("confidence").greater_than(
                        MIN_CONFIDENCE_SEMANTIC
                    ),
                )
                for obj in resp.objects:
                    results.append(RawMemory(
                        postgres_id=str(obj.properties.get("postgres_id", "")),
                        memory_type="semantic",
                        content=obj.properties.get("fact", ""),
                        retrieval_score=obj.metadata.score or 0.0,
                        properties=dict(obj.properties),
                        importance_score=float(obj.properties.get("importance_score", 0.5)),
                        confidence=float(obj.properties.get("confidence", 0.5)),
                    ))
            finally:
                client.close()
        except Exception:
            continue

    return results


def _query_procedural(
    agent_id: str,
    query_text: str,
    query_vector: list[float],
    task_type: Optional[str],
) -> Optional[RawMemory]:
    from weaviate.classes.query import MetadataQuery, HybridFusion, Filter

    try:
        client, col = _get_weaviate_collection("ProceduralMemory", agent_id)
        try:
            filters = None
            if task_type:
                filters = Filter.by_property("task_type").equal(task_type)

            resp = col.query.hybrid(
                query=query_text,
                vector=query_vector if query_vector else None,
                alpha=ALPHA_PROCEDURAL,
                limit=3,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=MetadataQuery(score=True),
                filters=filters,
            )

            if not resp.objects:
                return None

            best = max(
                resp.objects,
                key=lambda o: float(o.properties.get("confidence", 0.0)),
            )
            conf = float(best.properties.get("confidence", 0.0))
            if conf < MIN_CONFIDENCE_PROCEDURAL:
                return None

            return RawMemory(
                postgres_id=str(best.properties.get("postgres_id", "")),
                memory_type="procedural",
                content=best.properties.get("trigger_condition", ""),
                retrieval_score=best.metadata.score or 0.0,
                properties=dict(best.properties),
                importance_score=float(best.properties.get("importance_score", 0.5)),
                confidence=conf,
            )
        finally:
            client.close()

    except Exception:
        return None


def _query_episodic(
    agent_id: str,
    query_text: str,
    query_vector: list[float],
    task_type: Optional[str],
) -> list[RawMemory]:
    from weaviate.classes.query import MetadataQuery, HybridFusion, Filter

    try:
        client, col = _get_weaviate_collection("EpisodicMemory", agent_id)
        try:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(days=EPISODIC_DAYS_BACK)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")

            date_filter = Filter.by_property("session_start").greater_than(cutoff)
            combined = date_filter
            if task_type:
                combined = date_filter & Filter.by_property("task_type").equal(task_type)

            resp = col.query.hybrid(
                query=query_text,
                vector=query_vector if query_vector else None,
                alpha=ALPHA_EPISODIC,
                limit=K_EPISODIC + 2,
                fusion_type=HybridFusion.RELATIVE_SCORE,
                return_metadata=MetadataQuery(score=True),
                filters=combined,
            )

            results = []
            for obj in resp.objects:
                results.append(RawMemory(
                    postgres_id=str(obj.properties.get("postgres_id", "")),
                    memory_type="episodic",
                    content=obj.properties.get("task_prompt", ""),
                    retrieval_score=obj.metadata.score or 0.0,
                    properties=dict(obj.properties),
                    importance_score=float(obj.properties.get("importance_score", 0.5)),
                ))
            return results

        finally:
            client.close()

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_queries(
    task_prompt: str,
    agent_id: str,
    task_type: Optional[str] = None,
) -> QueryResult:
    """
    Embeds task_prompt and runs all three Weaviate queries.
    Fail-open on every step.
    """
    # 1. Embed
    t_embed = time.monotonic()
    vector, cache_hit = embed_text(task_prompt)
    embed_ms = int((time.monotonic() - t_embed) * 1000)

    # 2. Run queries
    t_query = time.monotonic()
    semantic   = _query_semantic(agent_id, task_prompt, vector)
    procedural = _query_procedural(agent_id, task_prompt, vector, task_type)
    episodic   = _query_episodic(agent_id, task_prompt, vector, task_type)
    query_ms   = int((time.monotonic() - t_query) * 1000)

    return QueryResult(
        semantic=semantic,
        procedural=procedural,
        episodic=episodic,
        embed_ms=embed_ms,
        query_ms=query_ms,
        cache_hit=cache_hit,
    )