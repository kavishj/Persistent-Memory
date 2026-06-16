"""
core/lifecycle/reconciler.py

Weaviate ↔ Postgres sync gap detection and repair.
Spec Day 3: reconciliation worker.

Postgres is authoritative. Weaviate is derived search index.

Decision table (spec Day 3 — validated):
  sync_status=pending,     age < 5min      → skip (still in flight)
  sync_status=pending,     5min–1hr        → re-queue embed job
  sync_status=pending,     > 1hr           → direct re-embed + insert
  sync_status=sync_failed, < 3 days        → retry each cycle
  sync_status=sync_failed, > 3 days        → critical alert + manual review flag
  In Weaviate but not Postgres             → delete from Weaviate (orphan cleanup)

Run schedule: every 24hr (Celery beat).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (spec Day 3)
# ---------------------------------------------------------------------------
PENDING_SKIP_MINUTES     = 5        # < 5 min: still in flight
PENDING_REQUEUE_MINUTES  = 60       # 5–60 min: re-queue
PENDING_DIRECT_HOURS     = 1        # > 1hr: direct re-embed
FAILED_RETRY_DAYS        = 3        # < 3 days: retry each cycle
FAILED_CRITICAL_DAYS     = 3        # >= 3 days: critical alert


# ---------------------------------------------------------------------------
# Repair action enum
# ---------------------------------------------------------------------------
class RepairAction(str, Enum):
    SKIP           = "skip"            # too new, still in flight
    REQUEUE        = "requeue"         # re-queue embed job
    DIRECT_EMBED   = "direct_embed"    # embed + insert directly
    RETRY          = "retry"           # retry sync_failed < 3d
    CRITICAL_ALERT = "critical_alert"  # sync_failed >= 3d, needs human
    DELETE_ORPHAN  = "delete_orphan"   # in Weaviate, not in Postgres


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ReconcileDecision:
    memory_id:    str                  # Postgres UUID (or Weaviate UUID for orphans)
    action:       RepairAction
    reason:       str
    sync_status:  Optional[str]        # pending|sync_failed|synced|None (orphan)
    age_minutes:  float


@dataclass
class ReconciliationReport:
    run_at:           datetime
    total_checked:    int
    skipped:          int
    requeued:         int
    direct_embedded:  int
    retried:          int
    critical_alerts:  int
    orphans_deleted:  int
    decisions:        list[ReconcileDecision] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------
def _decide_pending(memory_id: str, age_minutes: float) -> ReconcileDecision:
    """Determine repair action for sync_status=pending."""
    if age_minutes < PENDING_SKIP_MINUTES:
        return ReconcileDecision(
            memory_id=memory_id,
            action=RepairAction.SKIP,
            reason=f"pending {age_minutes:.1f}min < {PENDING_SKIP_MINUTES}min floor",
            sync_status="pending",
            age_minutes=age_minutes,
        )
    if age_minutes <= PENDING_REQUEUE_MINUTES:
        return ReconcileDecision(
            memory_id=memory_id,
            action=RepairAction.REQUEUE,
            reason=f"pending {age_minutes:.1f}min — re-queue embed job",
            sync_status="pending",
            age_minutes=age_minutes,
        )
    # > 1 hr
    return ReconcileDecision(
        memory_id=memory_id,
        action=RepairAction.DIRECT_EMBED,
        reason=f"pending {age_minutes:.1f}min > {PENDING_REQUEUE_MINUTES}min — direct re-embed",
        sync_status="pending",
        age_minutes=age_minutes,
    )


def _decide_failed(memory_id: str, age_minutes: float) -> ReconcileDecision:
    """Determine repair action for sync_status=sync_failed."""
    age_days = age_minutes / 1440
    if age_days < FAILED_RETRY_DAYS:
        return ReconcileDecision(
            memory_id=memory_id,
            action=RepairAction.RETRY,
            reason=f"sync_failed {age_days:.1f}d < {FAILED_RETRY_DAYS}d — retry",
            sync_status="sync_failed",
            age_minutes=age_minutes,
        )
    return ReconcileDecision(
        memory_id=memory_id,
        action=RepairAction.CRITICAL_ALERT,
        reason=f"sync_failed {age_days:.1f}d >= {FAILED_CRITICAL_DAYS}d — needs manual review",
        sync_status="sync_failed",
        age_minutes=age_minutes,
    )


def decide_repair(
    memory_id:   str,
    sync_status: str,
    updated_at:  datetime,
) -> ReconcileDecision:
    """
    Given a memory row from Postgres with sync gap, decide repair action.
    Only call for rows where sync_status in (pending, sync_failed).
    """
    now = datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    age_minutes = (now - updated_at).total_seconds() / 60

    if sync_status == "pending":
        return _decide_pending(memory_id, age_minutes)
    if sync_status == "sync_failed":
        return _decide_failed(memory_id, age_minutes)

    # Unexpected status — log and skip
    logger.warning(
        "Reconciler: unexpected sync_status=%s for memory_id=%s",
        sync_status, memory_id,
    )
    return ReconcileDecision(
        memory_id=memory_id,
        action=RepairAction.SKIP,
        reason=f"unexpected sync_status={sync_status}",
        sync_status=sync_status,
        age_minutes=age_minutes,
    )


def decide_orphan(weaviate_uuid: str) -> ReconcileDecision:
    """
    Memory exists in Weaviate but not Postgres → delete from Weaviate.
    Postgres is authoritative.
    """
    return ReconcileDecision(
        memory_id=weaviate_uuid,
        action=RepairAction.DELETE_ORPHAN,
        reason="in Weaviate but not in Postgres — orphan cleanup",
        sync_status=None,
        age_minutes=0,
    )


# ---------------------------------------------------------------------------
# Batch reconciler — called by Celery worker
# ---------------------------------------------------------------------------
def reconcile_batch(
    gap_rows:      list[dict],
    orphan_uuids:  list[str],
) -> ReconciliationReport:
    """
    Process a reconciliation batch.

    gap_rows: Postgres rows with sync_status in (pending, sync_failed).
      Each dict: {id, sync_status, updated_at, memory_type, agent_id}
    orphan_uuids: Weaviate UUIDs not found in Postgres.

    Returns ReconciliationReport.
    Caller (workers/tasks.py) is responsible for executing repair actions:
      REQUEUE      → push embed job to Celery queue
      DIRECT_EMBED → call embedding + Weaviate insert inline
      RETRY        → re-attempt Weaviate upsert
      CRITICAL_ALERT → write to memory_health_reports with severity=critical
      DELETE_ORPHAN → call weaviate_client.data_object.delete(uuid)
    """
    report = ReconciliationReport(
        run_at=datetime.now(timezone.utc),
        total_checked=len(gap_rows) + len(orphan_uuids),
        skipped=0,
        requeued=0,
        direct_embedded=0,
        retried=0,
        critical_alerts=0,
        orphans_deleted=0,
    )

    # Process Postgres gap rows
    for row in gap_rows:
        decision = decide_repair(
            memory_id=str(row["id"]),
            sync_status=row.get("sync_status", "pending"),
            updated_at=row.get("updated_at", datetime.now(timezone.utc)),
        )
        report.decisions.append(decision)

        if decision.action == RepairAction.SKIP:
            report.skipped += 1
        elif decision.action == RepairAction.REQUEUE:
            report.requeued += 1
        elif decision.action == RepairAction.DIRECT_EMBED:
            report.direct_embedded += 1
        elif decision.action == RepairAction.RETRY:
            report.retried += 1
        elif decision.action == RepairAction.CRITICAL_ALERT:
            report.critical_alerts += 1
            logger.critical(
                "Reconciler: CRITICAL sync_failed memory_id=%s — manual review required",
                decision.memory_id,
            )

    # Process Weaviate orphans
    for wuuid in orphan_uuids:
        decision = decide_orphan(wuuid)
        report.decisions.append(decision)
        report.orphans_deleted += 1
        logger.warning(
            "Reconciler: orphan Weaviate object uuid=%s — will delete", wuuid
        )

    logger.info(
        "Reconciler: checked=%d skip=%d requeue=%d direct=%d retry=%d critical=%d orphans=%d",
        report.total_checked,
        report.skipped,
        report.requeued,
        report.direct_embedded,
        report.retried,
        report.critical_alerts,
        report.orphans_deleted,
    )

    return report


# ---------------------------------------------------------------------------
# Weaviate → Postgres gap finder helper
# ---------------------------------------------------------------------------
def find_orphaned_weaviate_objects(
    weaviate_uuids: set[str],
    postgres_weaviate_ids: set[str],
) -> list[str]:
    """
    Returns Weaviate UUIDs that have no matching row in Postgres.
    weaviate_uuids: all UUIDs from Weaviate collection scan
    postgres_weaviate_ids: weaviate_id column from memory_entries (non-null)
    """
    orphans = list(weaviate_uuids - postgres_weaviate_ids)
    if orphans:
        logger.info("Reconciler: found %d orphaned Weaviate objects", len(orphans))
    return orphans