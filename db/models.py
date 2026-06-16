"""
db/models.py

SQLAlchemy ORM models.
Mirrors ddl.sql exactly — all column renames applied.
Spec Day 1.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime,
    Float, ForeignKey, Integer, Text, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# agents
# ---------------------------------------------------------------------------
class Agent(Base):
    __tablename__ = "agents"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_key    = Column(Text, nullable=False, unique=True)
    display_name = Column(Text, nullable=False)
    scope_tier   = Column(Text, nullable=False, default="standard")
    is_active    = Column(Boolean, nullable=False, default=True)
    api_key_hash = Column(Text, nullable=False)
    created_at   = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at   = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    metadata_    = Column("metadata", JSONB, default=dict)

    sessions = relationship("Session",     back_populates="agent")
    memories = relationship("MemoryEntry", back_populates="agent")


# ---------------------------------------------------------------------------
# task_types
# ---------------------------------------------------------------------------
class TaskType(Base):
    __tablename__ = "task_types"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name        = Column(Text, nullable=False, unique=True)
    description = Column(Text)
    created_at  = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# sessions
# ---------------------------------------------------------------------------
class Session(Base):
    __tablename__ = "sessions"

    id                 = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id           = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="RESTRICT"), nullable=False)
    task_type_id       = Column(UUID(as_uuid=True), ForeignKey("task_types.id"))
    task_prompt        = Column(Text, nullable=False, default="")
    task_prompt_tokens = Column(Integer)
    final_output       = Column(Text)
    outcome            = Column(Text)           # success | failure | partial
    error_message      = Column(Text)
    duration_ms        = Column(Integer)
    token_cost         = Column(Integer)
    memories_injected  = Column(ARRAY(UUID(as_uuid=True)))
    session_start      = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    session_end        = Column(DateTime(timezone=True))
    status             = Column(Text, nullable=False, default="active")
    metadata_          = Column("metadata", JSONB, default=dict)

    agent = relationship("Agent", back_populates="sessions")


# ---------------------------------------------------------------------------
# memory_entries
# ---------------------------------------------------------------------------
class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    # Identity
    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id    = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="RESTRICT"))
    scope       = Column(Text, nullable=False, default="agent")
    memory_type = Column(Text, nullable=False)   # episodic | semantic | procedural
    fact_type   = Column(Text)                   # constraint|preference|environment|capability|relationship

    # Content
    content        = Column(Text, nullable=False)
    content_tokens = Column(Integer)
    entities       = Column(ARRAY(Text))

    # Weaviate reference
    weaviate_class    = Column(Text)
    weaviate_id       = Column(UUID(as_uuid=True))
    sync_status       = Column(Text, nullable=False, default="pending")
    sync_priority     = Column(Text, nullable=False, default="normal")
    last_sync_attempt = Column(DateTime(timezone=True))

    # Lifecycle scores
    importance_score    = Column(Float, nullable=False, default=0.5)
    base_importance     = Column(Float, nullable=False, default=0.5)
    confidence          = Column(Float, nullable=False, default=0.5)
    access_count        = Column(Integer, nullable=False, default=0)
    successful_uses     = Column(Integer, nullable=False, default=0)
    total_uses          = Column(Integer, nullable=False, default=0)
    explicit_importance = Column(Float, nullable=False, default=0.5)

    # Timestamps
    created_at     = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    last_accessed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    ttl_expires    = Column(DateTime(timezone=True))
    deleted_at     = Column(DateTime(timezone=True))

    # Flags
    is_active           = Column(Boolean, nullable=False, default=True)
    pinned              = Column(Boolean, nullable=False, default=False)
    conflict_tag        = Column(Text, nullable=False, default="none")
    needs_manual_review = Column(Boolean, nullable=False, default=False)
    dedup_checked_at    = Column(DateTime(timezone=True))
    conflict_checked_at = Column(DateTime(timezone=True))

    # Relationships (FK)
    session_id         = Column(UUID(as_uuid=True), ForeignKey("sessions.id",        ondelete="SET NULL"))
    supersedes         = Column(UUID(as_uuid=True), ForeignKey("memory_entries.id",   ondelete="SET NULL"))
    superseded_by      = Column(UUID(as_uuid=True), ForeignKey("memory_entries.id",   ondelete="SET NULL"))
    linked_conflict_id = Column(UUID(as_uuid=True), ForeignKey("memory_entries.id",   ondelete="SET NULL"))

    # Episodic-specific
    outcome_feedback  = Column(Text)
    ep_session_start  = Column(DateTime(timezone=True))
    ep_is_summarized  = Column(Boolean, nullable=False, default=False)
    ep_summarized_at  = Column(DateTime(timezone=True))

    # Procedural-specific
    proc_task_type      = Column(Text)
    proc_version        = Column(Integer, default=1)
    proc_success_count  = Column(Integer, default=0)
    proc_failure_count  = Column(Integer, default=0)
    proc_last_used      = Column(DateTime(timezone=True))

    # Type-specific JSONB blob
    detail = Column(JSONB, nullable=False, default=dict)

    agent = relationship("Agent", back_populates="memories")


# ---------------------------------------------------------------------------
# memory_access_log
# ---------------------------------------------------------------------------
class MemoryAccessLog(Base):
    __tablename__ = "memory_access_log"

    id               = Column(BigInteger, primary_key=True, autoincrement=True)
    memory_id        = Column(UUID(as_uuid=True), ForeignKey("memory_entries.id", ondelete="CASCADE"), nullable=False)
    session_id       = Column(UUID(as_uuid=True), ForeignKey("sessions.id",       ondelete="SET NULL"))
    agent_id         = Column(UUID(as_uuid=True), ForeignKey("agents.id",         ondelete="SET NULL"))
    accessed_at      = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    outcome_recorded = Column(Text)
    retrieval_rank   = Column(Integer)
    retrieval_score  = Column(Float)


# ---------------------------------------------------------------------------
# memory_health_reports
# ---------------------------------------------------------------------------
class MemoryHealthReport(Base):
    __tablename__ = "memory_health_reports"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id    = Column(UUID(as_uuid=True), ForeignKey("agents.id", ondelete="CASCADE"))
    report_type = Column(Text, nullable=False, default="daily_summary")
    severity    = Column(Text, nullable=False, default="info")
    details     = Column(JSONB, nullable=False, default=dict)
    created_at  = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# deletion_audit_log
# ---------------------------------------------------------------------------
class DeletionAuditLog(Base):
    __tablename__ = "deletion_audit_log"

    id              = Column(BigInteger, primary_key=True, autoincrement=True)
    memory_id       = Column(UUID(as_uuid=True), nullable=False)
    agent_id        = Column(UUID(as_uuid=True))
    memory_type     = Column(Text)
    deleted_at      = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    hard_deleted_at = Column(DateTime(timezone=True))
    reason          = Column(Text)
    snapshot        = Column(JSONB, default=dict)
