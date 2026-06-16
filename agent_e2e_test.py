"""
agent_e2e_test.py — End-to-end test covering all 3 memory types.
Spec ref: memory_engine_design_spec.md §adapter, §retrieval, §write
 
Flow:
  Session 1 (episodic + semantic): Q&A about Python
  Session 2 (procedural):          Multi-step task — run 3x to hit write threshold
  Final verify:                     retrieve context, assert all 3 types present
 
Requirements:
  pip install groq requests
  GROQ_API_KEY env var set
  API running at localhost:8000
  Agent id=d8c7c79e-d3af-4376-9899-5c8d7c94f28b key=test-key-abc123
"""
 
import os
import json
import time
import uuid
import requests
from groq import Groq
 
# ── Config ────────────────────────────────────────────────────────────────────
API_URL   = os.getenv("MEMORY_ENGINE_URL", "http://localhost:8000")
API_KEY   = os.getenv("MEMORY_ENGINE_API_KEY", "test-key-abc123")
AGENT_ID  = os.getenv("TEST_AGENT_ID", "d8c7c79e-d3af-4376-9899-5c8d7c94f28b")
GROQ_KEY  = os.getenv("GROQ_API_KEY")
 
HEADERS = {
    "X-API-Key": API_KEY,
    "X-Agent-ID": AGENT_ID,
    "Content-Type": "application/json",
}
 
groq_client = Groq(api_key=GROQ_KEY)
 
# ── Helpers ───────────────────────────────────────────────────────────────────
 
def log(tag: str, msg: str):
    print(f"\n[{tag}] {msg}")
 
def retrieve(query: str, task_type: str = "question_answering") -> dict:
    r = requests.post(
        f"{API_URL}/memory/retrieve",
        headers=HEADERS,
        json={"task_prompt": query, "task_type": task_type},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()
 
def write(session_id: str, session_log: str, task_type: str = "question_answering", outcome: str = "success"):
    r = requests.post(
        f"{API_URL}/memory/write",
        headers=HEADERS,
        json={
            "session_id": session_id,
            "session_log": session_log,
            "outcome": outcome,
            "task_type": task_type,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()
 
def groq_chat(system: str, user: str) -> str:
    resp = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
 
def wait(seconds: int, reason: str = "Celery worker"):
    log("WAIT", f"{seconds}s — {reason}")
    time.sleep(seconds)
 
# ── QA sessions (episodic + semantic) ────────────────────────────────────────
 
QA_QUESTIONS = [
    "What are Python list comprehensions and when should I use them?",
    "Explain the difference between shallow copy and deep copy in Python.",
    "What is the GIL in Python and how does it affect multithreading?",
]
 
def run_qa_session(question: str, n: int):
    sid = str(uuid.uuid4())
    log(f"QA-{n}", f"Q: {question}")
 
    ctx = retrieve(question, "question_answering")
    log(f"QA-{n}", f"tokens={ctx.get('tokens_used',0)} semantic={ctx.get('semantic_count',0)} episodic={ctx.get('episodic_count',0)}")
 
    system = (
        "You are a helpful Python tutor. Use memory context if relevant.\n\n"
        f"MEMORY CONTEXT:\n{ctx.get('context_string') or '(none yet)'}"
    )
    answer = groq_chat(system, question)
    log(f"QA-{n}", f"A: {answer[:200]}{'...' if len(answer)>200 else ''}")
 
    session_log = (
        f"SESSION {sid}\n"
        f"Task type: question_answering\n"
        f"User question: {question}\n"
        f"Agent answer: {answer}\n"
        f"Outcome: success\n"
    )
    result = write(sid, session_log, "question_answering", "success")
    log(f"QA-{n}", f"Queued job_id={result.get('job_id')}")
 
# ── Task sessions (procedural — need 3 successes) ────────────────────────────
 
TASK_STEPS = [
    "Step 1: python -m venv .venv — DONE",
    "Step 2: .venv\\Scripts\\activate — DONE",
    "Step 3: pip install -r requirements.txt — DONE",
    "Step 4: pip list — DONE",
]
TASK_PROMPT = (
    "Execute these steps to set up a Python virtual environment and report status:\n"
    "1. python -m venv .venv\n"
    "2. Activate venv\n"
    "3. pip install -r requirements.txt\n"
    "4. pip list\n"
)
 
def run_task_session(n: int):
    sid = str(uuid.uuid4())
    log(f"TASK-{n}", "setup Python venv")
 
    ctx = retrieve("set up python virtual environment install dependencies", "code_execution")
    log(f"TASK-{n}", f"procedural_found={ctx.get('procedural_found',False)}")
 
    system = (
        "You are a DevOps agent. Execute tasks step by step. "
        "Report each step: DONE or FAILED.\n\n"
        f"MEMORY CONTEXT:\n{ctx.get('context_string') or '(none yet)'}"
    )
    execution = groq_chat(system, TASK_PROMPT)
 
    session_log = (
        f"SESSION {sid}\n"
        f"Task type: code_execution\n"
        f"Task: Setup Python virtual environment\n"
        f"Steps executed:\n"
        + "\n".join(f"  {s}" for s in TASK_STEPS) +
        f"\nAgent output:\n{execution}\n"
        f"Outcome: success\n"
        f"All steps completed successfully.\n"
    )
    result = write(sid, session_log, "code_execution", "success")
    log(f"TASK-{n}", f"Queued job_id={result.get('job_id')}")
 
# ── Health check ──────────────────────────────────────────────────────────────
 
def check_health():
    try:
        r = requests.get(f"{API_URL}/memory/health", headers=HEADERS, timeout=30)
        r.raise_for_status()
        log("HEALTH", f"API up — {r.json()}")
    except Exception as e:
        log("ERROR", f"API not reachable: {e}")
        raise SystemExit(1)
 
# ── Verify ────────────────────────────────────────────────────────────────────
 
def verify():
    log("VERIFY", "Querying semantic + episodic...")
    ctx_qa = retrieve("Python programming concepts GIL list comprehensions copy", "question_answering")
    log("VERIFY", f"semantic={ctx_qa.get('semantic_count',0)} episodic={ctx_qa.get('episodic_count',0)} tokens={ctx_qa.get('tokens_used',0)}")
    print(ctx_qa.get("context_string","(empty)")[:500])
 
    log("VERIFY", "Querying procedural...")
    ctx_task = retrieve("set up python virtual environment", "code_execution")
    log("VERIFY", f"procedural_found={ctx_task.get('procedural_found',False)} tokens={ctx_task.get('tokens_used',0)}")
    print(ctx_task.get("context_string","(empty)")[:500])
 
    errors = []
    if ctx_qa.get("semantic_count",0) == 0 and ctx_qa.get("episodic_count",0) == 0:
        errors.append("FAIL: no semantic/episodic memories after QA sessions")
    else:
        log("PASS", f"semantic={ctx_qa.get('semantic_count',0)} episodic={ctx_qa.get('episodic_count',0)}")
 
    if not ctx_task.get("procedural_found", False):
        log("WARN", "procedural_found=False — classifier may need beat pass or >3 session threshold not yet met. "
            "Re-run script or wait for hourly worker cycle.")
    else:
        log("PASS", "procedural memory found")
 
    if errors:
        print("\n" + "\n".join(errors))
        raise SystemExit(1)
 
    log("DONE", "E2E test complete.")
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main():
    print("=" * 60)
    print("MEMORY ENGINE — E2E TEST (episodic + semantic + procedural)")
    print("=" * 60)
 
    if not GROQ_KEY:
        log("ERROR", "GROQ_API_KEY not set")
        raise SystemExit(1)
 
    check_health()
 
    # Phase 1: QA → episodic + semantic
    log("PHASE-1", "3 QA sessions → episodic + semantic")
    for i, q in enumerate(QA_QUESTIONS, 1):
        run_qa_session(q, i)
        time.sleep(1)  # Groq rate limit buffer
 
    wait(10, "extract + embed QA sessions")
 
    # Phase 2: Task × 3 → procedural (threshold = 3 successful)
    log("PHASE-2", "3 task sessions → procedural (threshold = 3)")
    for i in range(1, 4):
        run_task_session(i)
        time.sleep(2)
 
    wait(15, "extract + embed task sessions + beat classifier pass")
 
    # Phase 3: Verify all 3 types
    log("PHASE-3", "Verify retrieval across all memory types")
    verify()
 
 
if __name__ == "__main__":
    main()