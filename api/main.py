"""
api/main.py

FastAPI application — 4 endpoints + API key auth middleware.
Spec Day 4.

Endpoints:
  POST /memory/retrieve  → context_string + metadata   SLA <200ms p95
  POST /memory/write     → queued: true + job_id        SLA <50ms
  GET  /memory/health    → latest health report
  DELETE /memory/expire  → operator tier only

Auth: API key in X-API-Key header → resolves (agent_id, scope_tier).
Scope tiers:
  standard:  read own + global
  elevated:  read own + global, write global
  operator:  full access + hard delete (manual creation only)

Write path: always async (Celery task), never blocks.
Retrieve path: sync, fail-open on every external call.
"""
from dotenv import load_dotenv
load_dotenv()
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Memory Engine API",
    version="1.0.0",
    description="Persistent memory engine for AI agents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
_engine = None
_SessionLocal = None


def _get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        db_url = os.environ.get(
            "DATABASE_URL",
            "postgresql://memory:memory@localhost:5432/memory_engine"
        )
        _engine = create_engine(db_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine)
    return _engine


def get_db():
    _get_engine()
    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Auth — API key → (agent_id, scope_tier)
# ---------------------------------------------------------------------------
SCOPE_TIERS = ("standard", "elevated", "operator")


class AuthContext:
    def __init__(self, agent_id: str, scope_tier: str):
        self.agent_id  = agent_id
        self.scope_tier = scope_tier

    @property
    def is_operator(self) -> bool:
        return self.scope_tier == "operator"

    @property
    def can_write_global(self) -> bool:
        return self.scope_tier in ("elevated", "operator")


def resolve_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> AuthContext:
    """
    Resolve X-API-Key header → AuthContext.
    Looks up agents table: api_key_hash + scope_tier.
    Raises 401 on missing/invalid key.
    Operator agents created manually only — no API endpoint.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header required",
        )

    try:
        # Use pgcrypto crypt() for hash comparison (stored as bcrypt hash)
        row = db.execute(text("""
            SELECT id, scope_tier
            FROM agents
            WHERE api_key_hash = crypt(:key, api_key_hash)
              AND is_active = TRUE
            LIMIT 1
        """), {"key": x_api_key}).fetchone()
    except Exception as e:
        logger.error("Auth DB error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service unavailable",
        )

    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )

    return AuthContext(agent_id=str(row.id), scope_tier=row.scope_tier)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

# POST /memory/retrieve
class RetrieveRequest(BaseModel):
    task_prompt:   str              = Field(..., min_length=1, max_length=8000)
    task_type:     Optional[str]    = Field(None, max_length=100)
    session_id:    Optional[str]    = Field(None)
    token_budget:  Optional[int]    = Field(None, ge=100, le=2000)

class RetrieveResponse(BaseModel):
    context_string:     str
    memory_ids_used:    list[str]
    session_id:         str
    tokens_used:        int
    procedural_found:   bool
    semantic_count:     int
    episodic_count:     int


# POST /memory/write
class WriteRequest(BaseModel):
    session_id:      str
    session_log:     str            = Field(..., min_length=1)   # serialized session
    outcome:         str            = Field(..., pattern="^(success|failure|partial)$")
    task_type:       Optional[str]  = Field(None, max_length=100)
    explicit_importance: Optional[float] = Field(None, ge=0.0, le=1.0)

class WriteResponse(BaseModel):
    queued:  bool
    job_id:  str


# GET /memory/health
class HealthResponse(BaseModel):
    agent_id:       str
    report_type:    str
    severity:       str
    details:        dict
    generated_at:   datetime


# DELETE /memory/expire
class ExpireRequest(BaseModel):
    memory_ids:  list[str]  = Field(..., min_items=1, max_items=500)
    reason:      str        = Field(..., min_length=1, max_length=500)

class ExpireResponse(BaseModel):
    expired:   int
    audit_ids: list[str]


# ---------------------------------------------------------------------------
# POST /memory/retrieve
# SLA: <200ms p95
# ---------------------------------------------------------------------------
@app.post("/memory/retrieve", response_model=RetrieveResponse)
async def retrieve_memory(
    req:  RetrieveRequest,
    auth: AuthContext = Depends(resolve_api_key),
    db:   Session     = Depends(get_db),
):
    """
    Retrieve relevant memory context for an agent task.
    Calls core/retrieval pipeline: embed → query → rerank → assemble.
    Fail-open: returns empty context on any retrieval error.
    """
    from core.retrieval.query_builder    import build_queries
    from core.retrieval.reranker         import rerank
    from core.retrieval.context_assembler import assemble_context

    session_id = req.session_id or str(uuid.uuid4())

    # Log session start (fail-open)
    try:
        db.execute(text("""
            INSERT INTO sessions (id, agent_id, session_start, status)
            VALUES (:id, :agent_id, NOW(), 'active')
            ON CONFLICT (id) DO NOTHING
        """), {"id": session_id, "agent_id": auth.agent_id})
        db.commit()
    except Exception as e:
        logger.warning("retrieve: session log failed (fail-open): %s", e)

    # Retrieval pipeline — fail-open at each stage
    try:
        results = build_queries(
            agent_id=auth.agent_id,
            task_prompt=req.task_prompt,
            task_type=req.task_type,
        )
    except Exception as e:
        logger.error("retrieve: query_builder failed (fail-open): %s", e)
        results = []

    try:
        ranked = rerank(results) if results else []
    except Exception as e:
        logger.error("retrieve: reranker failed (fail-open): %s", e)
        ranked = results

    try:
        token_budget = req.token_budget or 1500
        assembled = assemble_context(ranked)
    except Exception as e:
        logger.error("retrieve: context_assembler failed (fail-open): %s", e)
        assembled = type("Ctx", (), {
            "context_string":   "",
            "memory_ids_used":  [],
            "tokens_used":      0,
            "procedural_found": False,
            "semantic_count":   0,
            "episodic_count":   0,
        })()

    # Log accesses (fail-open)
    try:
        if assembled.memory_ids_used:
            for mid in assembled.memory_ids_used:
                db.execute(text("""
                    INSERT INTO memory_access_log
                        (id, memory_id, session_id, agent_id, accessed_at)
                    VALUES (gen_random_uuid(), :mid, :sid, :aid, NOW())
                """), {"mid": mid, "sid": session_id, "aid": auth.agent_id})
            db.commit()
    except Exception as e:
        logger.warning("retrieve: access log failed (fail-open): %s", e)

    return RetrieveResponse(
        context_string=assembled.context_string,
        memory_ids_used=assembled.memory_ids_used,
        session_id=session_id,
        tokens_used=assembled.tokens_used,
        procedural_found=assembled.procedural_found,
        semantic_count=assembled.semantic_count,
        episodic_count=assembled.episodic_count,
    )


# ---------------------------------------------------------------------------
# POST /memory/write
# SLA: <50ms — enqueue only, never block
# ---------------------------------------------------------------------------
@app.post("/memory/write", response_model=WriteResponse)
async def write_memory(
    req:  WriteRequest,
    auth: AuthContext = Depends(resolve_api_key),
    db:   Session     = Depends(get_db),
):
    """
    Enqueue session log for async memory extraction.
    Never blocks — returns job_id immediately.
    Write path: extract → classify → dedup → conflict → store (Celery).
    No mid-session writes — only called post-session.
    """
    # Scope check: standard agents can only write for themselves
    # elevated/operator can write global scope (handled in Celery task)
    job_id = str(uuid.uuid4())

    # Close session record
    try:
        db.execute(text("""
            UPDATE sessions
            SET status = 'completed',
                ended_at = NOW(),
                outcome = :outcome
            WHERE id = :sid AND agent_id = :aid
        """), {
            "sid":     req.session_id,
            "aid":     auth.agent_id,
            "outcome": req.outcome,
        })
        db.commit()
    except Exception as e:
        logger.warning("write: session close failed (fail-open): %s", e)

    # Enqueue Celery task (fail-open — log error, still return queued=True
    # so agent is never blocked)
    try:
        from workers.tasks import app as celery_app
        celery_app.send_task(
            "workers.tasks.process_write_job",
            kwargs={
                "job_id":               job_id,
                "agent_id":             auth.agent_id,
                "session_id":           req.session_id,
                "session_log":          req.session_log,
                "outcome":              req.outcome,
                "task_type":            req.task_type,
                "explicit_importance":  req.explicit_importance,
                "can_write_global":     auth.can_write_global,
            },
            task_id=job_id,
        )
    except Exception as e:
        logger.error("write: Celery enqueue failed (fail-open): %s", e)

    return WriteResponse(queued=True, job_id=job_id)


# ---------------------------------------------------------------------------
# GET /memory/health
# ---------------------------------------------------------------------------
@app.get("/memory/health", response_model=HealthResponse)
async def get_health(
    auth: AuthContext = Depends(resolve_api_key),
    db:   Session     = Depends(get_db),
):
    """
    Return latest health report for the authenticated agent.
    Operators can see any agent (future: add ?agent_id= param).
    """
    try:
        row = db.execute(text("""
            SELECT agent_id, report_type, severity, details, created_at
            FROM memory_health_reports
            WHERE agent_id = :aid
            ORDER BY created_at DESC
            LIMIT 1
        """), {"aid": auth.agent_id}).fetchone()
    except Exception as e:
        logger.error("health: DB error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health report unavailable",
        )

    if not row:
        # No report yet — return empty healthy state
        return HealthResponse(
            agent_id=auth.agent_id,
            report_type="daily_summary",
            severity="info",
            details={"message": "No health report generated yet"},
            generated_at=datetime.now(timezone.utc),
        )

    import json as _json
    details = row.details
    if isinstance(details, str):
        try:
            details = _json.loads(details)
        except Exception:
            details = {"raw": details}

    return HealthResponse(
        agent_id=str(row.agent_id),
        report_type=row.report_type,
        severity=row.severity,
        details=details,
        generated_at=row.created_at,
    )


# ---------------------------------------------------------------------------
# DELETE /memory/expire
# Operator tier only
# ---------------------------------------------------------------------------
@app.delete("/memory/expire", response_model=ExpireResponse)
async def expire_memories(
    req:  ExpireRequest,
    auth: AuthContext = Depends(resolve_api_key),
    db:   Session     = Depends(get_db),
):
    """
    Hard-expire specific memories immediately.
    Operator scope only.
    Writes deletion_audit_log entries before soft-delete.
    Actual hard delete happens in weekly worker after 7-day window.
    """
    if not auth.is_operator:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator scope required for manual expiry",
        )

    # Validate memory_ids belong to agent (or operator sees all)
    try:
        rows = db.execute(text("""
            SELECT id, agent_id, memory_type, content,
                   importance_score, created_at
            FROM memory_entries
            WHERE id = ANY(:ids)
              AND deleted_at IS NULL
        """), {"ids": req.memory_ids}).fetchall()
    except Exception as e:
        logger.error("expire: DB fetch failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable",
        )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active memories found for provided IDs",
        )

    import json as _json
    audit_ids  = []
    expire_ids = [str(row.id) for row in rows]

    # Write audit log entries
    for row in rows:
        audit_id = str(uuid.uuid4())
        db.execute(text("""
            INSERT INTO deletion_audit_log
                (id, memory_id, agent_id, memory_type,
                 deleted_at, hard_deleted_at, reason, snapshot)
            VALUES (
                :audit_id, :memory_id, :agent_id,
                :memory_type, NOW(), NULL,
                :reason,
                :snapshot
            )
        """), {
            "audit_id":   audit_id,
            "memory_id":  str(row.id),
            "agent_id":   str(row.agent_id),
            "memory_type": row.memory_type,
            "reason":     f"operator_manual: {req.reason}",
            "snapshot":   _json.dumps({
                "content_preview":  str(row.content or "")[:200],
                "importance_score": float(row.importance_score or 0),
            }),
        })
        audit_ids.append(audit_id)

    # Soft delete
    db.execute(text("""
        UPDATE memory_entries
        SET deleted_at = NOW(), updated_at = NOW()
        WHERE id = ANY(:ids)
    """), {"ids": expire_ids})

    db.commit()

    logger.info(
        "expire: operator=%s expired=%d reason=%r",
        auth.agent_id, len(expire_ids), req.reason,
    )

    return ExpireResponse(expired=len(expire_ids), audit_ids=audit_ids)


# ---------------------------------------------------------------------------
# Startup / health ping
# ---------------------------------------------------------------------------
@app.get("/ping")
async def ping():
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}



