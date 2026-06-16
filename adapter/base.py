"""
adapter/base.py

Pluggable adapter base class for existing agents.
Spec Day 4: ~30 min integration target.

Two changes to any existing agent:
  1. Subclass MemoryAdapter, implement 4 abstract methods
  2. Wrap run_task() with pre_task() / post_task()

All HTTP calls to memory engine are async + fail-open.
Agent is never blocked by memory failures.
"""

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default timeouts (ms → seconds)
# ---------------------------------------------------------------------------
RETRIEVE_TIMEOUT_S = 0.190   # <200ms SLA with 10ms headroom
WRITE_TIMEOUT_S    = 0.045   # <50ms SLA with 5ms headroom


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class RetrievedContext:
    """Returned by pre_task(). Passed to agent as context."""
    context_string:   str
    session_id:       str
    memory_ids_used:  list[str]
    tokens_used:      int
    procedural_found: bool
    semantic_count:   int
    episodic_count:   int
    retrieved_at:     float = field(default_factory=time.time)

    # Empty context sentinel — agent always gets a valid object
    @classmethod
    def empty(cls, session_id: str) -> "RetrievedContext":
        return cls(
            context_string="",
            session_id=session_id,
            memory_ids_used=[],
            tokens_used=0,
            procedural_found=False,
            semantic_count=0,
            episodic_count=0,
        )


@dataclass
class TaskResult:
    """Passed to post_task() after agent completes work."""
    session_id:          str
    output:              str
    outcome:             str              # success | failure | partial
    task_type:           Optional[str]    = None
    explicit_importance: Optional[float]  = None
    raw_session_log:     Optional[object] = None   # agent's raw log — serialized by adapter


# ---------------------------------------------------------------------------
# Abstract base adapter
# ---------------------------------------------------------------------------
class MemoryAdapter(ABC):
    """
    Base class for memory engine integration.
    Subclass and implement 4 abstract methods.
    All network calls are async + fail-open.
    """

    # ------------------------------------------------------------------
    # Abstract interface — implement these 4 methods
    # ------------------------------------------------------------------

    @abstractmethod
    def get_agent_id(self) -> str:
        """Return stable agent ID string (e.g. 'my-agent-01')."""
        ...

    @abstractmethod
    def get_api_key(self) -> str:
        """Return MEMORY_ENGINE_API_KEY for this agent."""
        ...

    @abstractmethod
    def get_engine_url(self) -> str:
        """Return base URL of memory engine (e.g. 'http://localhost:8000')."""
        ...

    @abstractmethod
    def serialize_session_log(self, raw: object) -> str:
        """
        Serialize the agent's raw session log to a JSON string.
        Called inside post_task() before writing to memory engine.
        Example: return json.dumps(raw, default=str)
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def get_task_type(self, task_prompt: str) -> Optional[str]:
        """
        Optionally extract task_type from prompt.
        Override for smarter classification.
        Default: None (classifier in extractor.py will determine it).
        """
        return None

    def get_retrieve_timeout(self) -> float:
        """Override to change retrieve timeout (seconds). Default: 190ms."""
        return RETRIEVE_TIMEOUT_S

    def get_write_timeout(self) -> float:
        """Override to change write timeout (seconds). Default: 45ms."""
        return WRITE_TIMEOUT_S

    # ------------------------------------------------------------------
    # Core integration methods — call these in your agent
    # ------------------------------------------------------------------

    async def pre_task(self, task_prompt: str) -> RetrievedContext:
        """
        Call at the START of each task.
        Retrieves relevant memory context from the engine.
        Always returns a RetrievedContext — never raises.

        Usage:
            ctx = await adapter.pre_task(task_prompt)
            full_prompt = f"{ctx.context_string}\\n\\n{task_prompt}"
        """
        session_id = str(uuid.uuid4())

        url     = f"{self.get_engine_url().rstrip('/')}/memory/retrieve"
        headers = self._auth_headers()
        payload = {
            "task_prompt": task_prompt,
            "task_type":   self.get_task_type(task_prompt),
            "session_id":  session_id,
        }

        try:
            async with httpx.AsyncClient(timeout=self.get_retrieve_timeout()) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            return RetrievedContext(
                context_string=data.get("context_string", ""),
                session_id=data.get("session_id", session_id),
                memory_ids_used=data.get("memory_ids_used", []),
                tokens_used=data.get("tokens_used", 0),
                procedural_found=data.get("procedural_found", False),
                semantic_count=data.get("semantic_count", 0),
                episodic_count=data.get("episodic_count", 0),
            )

        except httpx.TimeoutException:
            logger.warning(
                "MemoryAdapter.pre_task: timeout for agent=%s (fail-open)",
                self.get_agent_id(),
            )
        except httpx.HTTPStatusError as e:
            logger.warning(
                "MemoryAdapter.pre_task: HTTP %s for agent=%s (fail-open)",
                e.response.status_code, self.get_agent_id(),
            )
        except Exception as e:
            logger.error(
                "MemoryAdapter.pre_task: unexpected error agent=%s: %s (fail-open)",
                self.get_agent_id(), e,
            )

        # Fail-open: return empty context, agent proceeds without memory
        return RetrievedContext.empty(session_id)

    async def post_task(self, result: TaskResult) -> bool:
        """
        Call at the END of each task.
        Enqueues session log for async memory extraction.
        Returns True if enqueued, False on failure (non-blocking either way).

        Usage:
            await adapter.post_task(TaskResult(
                session_id=ctx.session_id,
                output=result,
                outcome="success",
            ))
        """
        url     = f"{self.get_engine_url().rstrip('/')}/memory/write"
        headers = self._auth_headers()

        # Serialize session log
        try:
            session_log = self.serialize_session_log(result.raw_session_log or result.output)
        except Exception as e:
            logger.warning(
                "MemoryAdapter.post_task: serialize failed agent=%s: %s — using output string",
                self.get_agent_id(), e,
            )
            session_log = json.dumps({"output": str(result.output)[:4000]})

        payload = {
            "session_id":          result.session_id,
            "session_log":         session_log,
            "outcome":             result.outcome,
            "task_type":           result.task_type or self.get_task_type(""),
            "explicit_importance": result.explicit_importance,
        }

        try:
            async with httpx.AsyncClient(timeout=self.get_write_timeout()) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                return data.get("queued", False)

        except httpx.TimeoutException:
            logger.warning(
                "MemoryAdapter.post_task: timeout agent=%s (fail-open)",
                self.get_agent_id(),
            )
        except httpx.HTTPStatusError as e:
            logger.warning(
                "MemoryAdapter.post_task: HTTP %s agent=%s (fail-open)",
                e.response.status_code, self.get_agent_id(),
            )
        except Exception as e:
            logger.error(
                "MemoryAdapter.post_task: unexpected error agent=%s: %s (fail-open)",
                self.get_agent_id(), e,
            )

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict:
        return {
            "X-API-Key":    self.get_api_key(),
            "Content-Type": "application/json",
        }


# ---------------------------------------------------------------------------
# Example implementation (reference — not imported by engine)
# ---------------------------------------------------------------------------
class ExampleAdapter(MemoryAdapter):
    """
    Minimal reference implementation.
    Copy this pattern into your agent — two changes to run_task().
    """

    def get_agent_id(self) -> str:
        return "my-agent-01"

    def get_api_key(self) -> str:
        return os.environ["MEMORY_ENGINE_API_KEY"]

    def get_engine_url(self) -> str:
        return os.environ.get("MEMORY_ENGINE_URL", "http://localhost:8000")

    def serialize_session_log(self, raw: object) -> str:
        return json.dumps(raw, default=str)


# ---------------------------------------------------------------------------
# Sync wrapper for non-async agents
# ---------------------------------------------------------------------------
class SyncMemoryAdapter(MemoryAdapter, ABC):
    """
    Sync wrapper for agents that can't use async.
    Runs pre_task/post_task in an asyncio event loop.
    """

    def pre_task_sync(self, task_prompt: str) -> RetrievedContext:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.pre_task(task_prompt))
                    return future.result(timeout=1.0)
            return loop.run_until_complete(self.pre_task(task_prompt))
        except Exception as e:
            logger.error("SyncMemoryAdapter.pre_task_sync failed: %s (fail-open)", e)
            return RetrievedContext.empty(str(uuid.uuid4()))

    def post_task_sync(self, result: TaskResult) -> bool:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.post_task(result))
                    return future.result(timeout=1.0)
            return loop.run_until_complete(self.post_task(result))
        except Exception as e:
            logger.error("SyncMemoryAdapter.post_task_sync failed: %s (fail-open)", e)
            return False
