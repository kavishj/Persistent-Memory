# Memory Engine

Self-hosted, open-source persistent memory engine for AI agents. Stores, indexes, and retrieves three memory types (episodic, semantic, procedural) scoped per agent, with vector-based similarity search, lifecycle management, and a pluggable adapter (~30 min to integrate).

---

## Table of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Quickstart — Docker](#quickstart--docker)
- [Quickstart — Manual](#quickstart--manual)
- [API Reference](#api-reference)
- [Adapter Integration Guide](#adapter-integration-guide)
- [Configuration Reference](#configuration-reference)
- [Memory Types](#memory-types)
- [Lifecycle & Scheduled Tasks](#lifecycle--scheduled-tasks)
- [Access Tiers](#access-tiers)
- [Numeric Constants](#numeric-constants)

---

## What It Does

Every AI agent session is ephemeral by default. Memory Engine gives agents persistent memory across sessions:

- **Before a task** — agent retrieves relevant context (facts, past procedures, past events) via `/memory/retrieve`
- **After a task** — agent submits the session log via `/memory/write`; the engine extracts, classifies, deduplicates, and indexes memories asynchronously
- **Background** — scheduled workers score importance, promote episodic → semantic, expire stale memories, and keep Weaviate + Postgres in sync

---

## Architecture

```
WRITE PATH (async)
Agent → POST /memory/write → Celery queue
  → extractor   (Groq llama-3.1-8b-instant)
  → classifier  (keyword + LLM fallback)
  → deduplicator (0.92 cosine threshold)
  → conflict_resolver
  → Postgres (system of record) + Weaviate (search index)

READ PATH (sync, <200ms p95)
Agent → POST /memory/retrieve
  → embed query (sentence-transformers/all-MiniLM-L6-v2, local)
  → 3 parallel Weaviate queries (semantic k=5, procedural k=1, episodic k=3)
  → Redis cache
  → reranker (retrieval×0.5 + importance×0.3 + recency×0.2)
  → context assembler (2000 token ceiling, hard slot reservation)
  → context string returned

LIFECYCLE PATH (scheduled)
Celery Beat → scorer → summarizer → expiry → reconciler
```

**System of record:** Postgres. Weaviate is a derived search index — always reconstructable from Postgres.

---

## Tech Stack

| Component | Choice |
|---|---|
| Vector DB | Weaviate 1.27.4 (self-hosted) |
| Relational DB | PostgreSQL 16 + pgvector |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384 dims, local, free) |
| Extraction LLM | Groq llama-3.1-8b-instant (primary), Anthropic claude-haiku-4-5 (optional) |
| API | FastAPI |
| Task queue | Celery + Redis |
| Language | Python 3.10 |
| Containers | Docker + Docker Compose |

---

## Quickstart — Docker

**Prerequisites:** Docker Desktop, `GROQ_API_KEY`.

```bash
# 1. Clone
git clone <repo> memory_engine
cd memory_engine

# 2. Create .env
echo "GROQ_API_KEY=your-key-here" > .env

# 3. Build + start everything
docker compose up --build

# 4. Verify
curl http://localhost:8000/memory/health \
  -H "X-API-Key: test-key-abc123" \
  -H "X-Agent-ID: d8c7c79e-d3af-4376-9899-5c8d7c94f28b"
```

Startup order is enforced automatically:
```
postgres → weaviate → redis → seed → worker → beat → api
```

All services health-checked before dependents start. On first boot, `seed` runs `scripts/seed_schema.py` to initialize Postgres tables and Weaviate collections, then exits.

---

## Quickstart — Manual

**Prerequisites:** Python 3.10, Docker Desktop (for Postgres/Weaviate/Redis), `GROQ_API_KEY`.

### Step 1 — Start infrastructure

```bash
docker compose up postgres weaviate redis -d
```

### Step 2 — Python environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 3 — Seed schema

```bash
DATABASE_URL=postgresql://memory:memory@localhost:5432/memory_engine \
  python scripts/seed_schema.py
```

### Step 4 — Three terminals

**Terminal 1 — API:**
```powershell
$env:DATABASE_URL = "postgresql://memory:memory@localhost:5432/memory_engine"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Worker:**
```powershell
celery -A workers.tasks worker --loglevel=info --pool=solo
```

**Terminal 3 — Beat:**
```powershell
celery -A workers.tasks beat --loglevel=info
```

### Step 5 — Verify

```bash
curl http://localhost:8000/memory/health \
  -H "X-API-Key: test-key-abc123" \
  -H "X-Agent-ID: d8c7c79e-d3af-4376-9899-5c8d7c94f28b"
```

---

## API Reference

All endpoints require:
```
X-API-Key: <agent-api-key>
X-Agent-ID: <agent-uuid>
```

### POST /memory/retrieve

Retrieve relevant memory context before a task. **SLA: <200ms p95.**

**Request:**
```json
{
  "task_prompt": "What is the best way to handle Python exceptions?",
  "task_type": "question_answering",
  "session_id": "optional-uuid-to-reuse",
  "token_budget": 1500
}
```

**Response:**
```json
{
  "context_string": "## Known Constraints & Facts\n- ...",
  "memory_ids_used": ["uuid1", "uuid2"],
  "session_id": "generated-or-provided-uuid",
  "tokens_used": 312,
  "procedural_found": false,
  "semantic_count": 5,
  "episodic_count": 0
}
```

Fail-open — returns empty context on any retrieval error. Agent never blocked.

---

### POST /memory/write

Submit session log for async processing after a task. **SLA: <50ms.**

**Request:**
```json
{
  "session_id": "uuid-from-retrieve",
  "session_log": "User asked about X. Agent answered Y. Outcome: success.",
  "outcome": "success",
  "task_type": "question_answering",
  "explicit_importance": 1
}
```

**Response:**
```json
{
  "queued": true,
  "job_id": "celery-task-uuid"
}
```

Returns immediately. Extraction, classification, dedup, and embedding happen asynchronously via Celery.

---

### GET /memory/health

Returns latest daily health report for the agent.

**Response:**
```json
{
  "agent_id": "uuid",
  "report_type": "daily_summary",
  "severity": "info",
  "details": { ... },
  "generated_at": "2026-06-17T00:00:00Z"
}
```

---

### DELETE /memory/expire

Trigger immediate expiry pass for the agent. **Requires operator tier.**

**Response:**
```json
{ "expired": 12 }
```

---

## Adapter Integration Guide

Integrate any existing agent in ~30 minutes with two changes.

### Step 1 — Subclass MemoryAdapter

```python
import os
import json
from adapter.base import MemoryAdapter

class MyAgentAdapter(MemoryAdapter):
    def get_agent_id(self) -> str:
        return "my-agent-01"

    def get_api_key(self) -> str:
        return os.environ["MEMORY_ENGINE_API_KEY"]

    def get_engine_url(self) -> str:
        return os.environ["MEMORY_ENGINE_URL"]

    def serialize_session_log(self, raw) -> str:
        return json.dumps(raw, default=str)
```

### Step 2 — Wrap your task function

```python
from adapter.base import TaskResult

_adapter = MyAgentAdapter()

async def run_task(task_prompt: str) -> str:
    # Pre-task: retrieve memory context
    ctx = await _adapter.pre_task(task_prompt)
    full_prompt = f"{ctx.context_string}\n\n{task_prompt}"

    # Your existing agent logic — unchanged
    try:
        result = await your_existing_agent(full_prompt)
        outcome = "success"
    except Exception as e:
        result, outcome = "", "failure"

    # Post-task: write session to memory
    await _adapter.post_task(TaskResult(
        session_id=ctx.session_id,
        output=result,
        outcome=outcome,
    ))

    return result
```

That's it. Two changes: subclass + wrap. No other modifications to your agent.

### Sync agents

```python
from adapter.base import SyncMemoryAdapter

class MyAdapter(SyncMemoryAdapter):
    ...

_adapter = MyAdapter()

def run_task(task_prompt: str) -> str:
    ctx = _adapter.pre_task_sync(task_prompt)
    result = your_agent(f"{ctx.context_string}\n\n{task_prompt}")
    _adapter.post_task_sync(TaskResult(
        session_id=ctx.session_id,
        output=result,
        outcome="success",
    ))
    return result
```

### Environment variables for adapter

```bash
MEMORY_ENGINE_URL=http://localhost:8000
MEMORY_ENGINE_API_KEY=your-agent-api-key
```

---

## Configuration Reference

All config via environment variables.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://memory:memory@localhost:5432/memory_engine` | Postgres connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `WEAVIATE_URL` | `http://localhost:8080` | Weaviate endpoint |
| `GROQ_API_KEY` | — | Groq API key (required for extraction) |
| `ANTHROPIC_API_KEY` | — | Anthropic key (optional, fallback extractor) |
| `MEMORY_ENGINE_API_KEY` | — | Key for adapter clients |
| `MEMORY_ENGINE_URL` | — | Engine URL for adapter clients |

---

## Memory Types

| Type | What it stores | Written when |
|---|---|---|
| **Semantic** | Extracted facts, constraints, preferences | Every successful session via extractor |
| **Episodic** | Raw session events, failures, anomalies | Failure outcomes or explicit event markers |
| **Procedural** | Step-by-step task procedures | After 3+ successful sessions of same task type |

### Context slot allocation (2000 token ceiling)

| Condition | Procedural | Semantic | Episodic |
|---|---|---|---|
| Procedural found | 600 | 600 | 300 |
| No procedural | — | 1000 | 500 |

---

## Lifecycle & Scheduled Tasks

Celery Beat runs 13 scheduled tasks:

| Schedule | Tasks |
|---|---|
| Every 5 min | `process_embed_queue`, `process_summarization_queue`, `apply_retrieval_bumps` |
| Every 1 hr | `recalculate_importance_hourly`, `run_deduplication_pass`, `detect_new_conflicts` |
| Every 24 hr | `full_importance_recalculation`, `soft_delete_expired`, `sync_reconciliation`, `generate_health_reports` |
| Every 7 days | `hard_delete_weekly`, `check_stale_procedural` |

**Soft delete:** memories marked deleted in hot path, hard deleted after 7-day window by weekly worker.

**Episodic → Semantic promotion:** summarizer condenses episodic clusters into semantic facts.

**Weaviate sync:** reconciler detects and repairs gaps between Postgres (source of record) and Weaviate (search index) every 24hr.

---

## Access Tiers

| Tier | Permissions | Created via |
|---|---|---|
| `standard` | Read own + global memories | API |
| `elevated` | Read own + global, write global | API |
| `operator` | Full access + hard delete + expire endpoint | Manual only |

Operator agents must be created manually in the database — no API endpoint exists by design.

---

## Numeric Constants

Do not change these without updating the full spec.

```
Reranker weights:    retrieval=0.50, importance=0.30, recency=0.20
Importance weights (episodic):   recency=0.50, access=0.20, outcome=0.20, explicit=0.10
Importance weights (semantic):   recency=0.25, access=0.30, outcome=0.30, explicit=0.15
Importance weights (procedural): recency=0.10, access=0.35, outcome=0.40, explicit=0.15
Base/dynamic split:  base=30%, dynamic=70%
Dedup threshold:     0.92 cosine similarity
Conflict supersede:  confidence >= 0.85
Confidence reduction: 0.15
Procedural threshold: 3 successful sessions minimum
Embedding dims:      384 (all-MiniLM-L6-v2)
Token ceiling:       2000 (target 1500)
HNSW ef:            semantic=128, procedural=64, episodic=64
HNSW efConstruction: semantic=256, procedural=256, episodic=128
TTL floors:          episodic raw=7d, semantic constraint=14d,
                     semantic preference=45d, procedural active=90d
```

---

## Running the E2E Test

Requires all services running + `GROQ_API_KEY` set.

```powershell
$env:GROQ_API_KEY = "your-key"
python agent_e2e_test.py
```

Runs 3 QA sessions (episodic + semantic) + 3 task sessions (procedural) + verifies retrieval across all memory types.
