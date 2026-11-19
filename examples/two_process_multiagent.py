"""Phase 5 Real Multi-Process Multi-Agent Example (Section 7).

Two independent OS processes (Researcher A using Qwen2.5-0.5B, Executor B using Llama-3.2-1B)
communicate strictly via Vect-Link SHM (VectLinkRing + KVArena) and CMCP envelopes,
synchronized via OS multiprocessing.Event (zero Sleep() calls).

Demonstrates all 9 required steps:
  1. A and B start independently.
  2. A creates and updates CMCP semantic state.
  3. B recovers state without A's private conversation history.
  4. B answers correctly using reconstructed CMCP context.
  5. A introduces a correction and invalidation.
  6. B receives the incremental update.
  7. B stops using obsolete data.
  8. B queries a historical version via checkout_seq().
  9. Consumer B restarts cleanly and recovers state from KVArena.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vectlink_cmcp import (  # noqa: E402
    CMCPAdapter,
    ConsumerSession,
    ProducerSession,
    StateOp,
    VectLinkArenaTransport,
    VectLinkRingTransport,
)


def _llama_query(model_path: str, prompt: str, max_tokens: int = 48) -> str:
    cmcp_root = Path(os.environ.get("CMCP_ROOT", ROOT.parent / "Cross-Model-Context-Protocol"))
    exe = cmcp_root / "build" / "cmcp_llama_harness.exe"
    env = os.environ.copy()
    env["PATH"] = r"C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin;" + env.get("PATH", "")
    proc = subprocess.Popen(
        [str(exe), "--mode", "persistent", "--model", model_path, "--n-ctx", "1024", "--threads", "4"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding="utf-8", env=env,
    )
    try:
        assert proc.stdout is not None and proc.stdin is not None
        _ = proc.stdout.readline()
        esc = prompt.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
        proc.stdin.write(f"{max_tokens}\t{esc}\n")
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline())
        return str(resp.get("output_text", "")).strip()
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=15)


def _process_a_researcher(
    ring_shm: str,
    arena_shm: str,
    model_a: str,
    ev_b_ready: Any,
    ev_msg1_sent: Any,
    ev_b_answered1: Any,
    ev_msg2_sent: Any,
    ev_done: Any,
    out_q: Any,
) -> None:
    ring = VectLinkRingTransport.create(ring_shm, capacity=8, max_payload_bytes=16384)
    arena = VectLinkArenaTransport.create(arena_shm, num_blocks=16, block_size_bytes=16384)
    try:
        assert ev_b_ready.wait(timeout=20.0)
        prod = ProducerSession(CMCPAdapter("multiagent-session"), ring)

        # Private conversation inside A (never sent raw to B)
        private_summary = _llama_query(
            model_a,
            "Summarize in 10 words: Primary node Node-700 uses route R-10 and budget 90ms.",
            max_tokens=20,
        )

        # Step 2: A publishes initial CMCP semantic state (seq 1..4)
        ops1 = [
            StateOp("e_node", "UPSERT", "entity", "Node-700", "primary", seq=1, version=1),
            StateOp("f_route", "UPSERT", "fact", "route", "R-10", seq=2, version=1),
            StateOp("c_budget", "UPSERT", "constraint", "budget", "budget<=90ms", seq=3, version=1),
            StateOp("f_tag", "UPSERT", "fact", "temp_tag", "TMP-OLD", seq=4, version=1),
        ]
        pkt1 = prod.commit_and_publish(ops1)
        ident1, _ = arena.publish_block(seq=4, payload=pkt1)
        ev_msg1_sent.set()

        assert ev_b_answered1.wait(timeout=30.0)

        # Step 5: A introduces a correction (route -> R-99, budget -> 35ms) and invalidates TMP-OLD
        ops2 = [
            StateOp("f_route", "UPSERT", "fact", "route", "R-99", seq=5, version=2),
            StateOp("c_budget", "UPSERT", "constraint", "budget", "budget<=35ms", seq=6, version=2),
            StateOp("f_tag", "INVALIDATE", "fact", "temp_tag", "", seq=7, version=2),
        ]
        pkt2 = prod.commit_and_publish(ops2)
        ident2, _ = arena.publish_block(seq=7, payload=pkt2)
        out_q.put({"role": "A", "private_summary": private_summary, "ident1": ident1, "ident2": ident2})
        ev_msg2_sent.set()

        assert ev_done.wait(timeout=30.0)
    finally:
        arena.close(unlink=True)
        ring.close(unlink=True)


def _process_b_executor(
    ring_shm: str,
    arena_shm: str,
    model_b: str,
    ev_b_ready: Any,
    ev_msg1_sent: Any,
    ev_b_answered1: Any,
    ev_msg2_sent: Any,
    out_q: Any,
) -> None:
    ring = VectLinkRingTransport.open(ring_shm)
    ev_b_ready.set()
    try:
        cons = ConsumerSession(CMCPAdapter("multiagent-session"), ring)

        # Step 3 & 4: Wait for msg1, reconstruct without A's private history, answer with Llama-3.2-1B
        assert ev_msg1_sent.wait(timeout=30.0)
        res1 = cons.poll_and_apply_ring()
        assert res1 is not None
        ctx1 = cons.render_context("r3")
        ans1 = _llama_query(
            model_b,
            f"Context:\n{ctx1}\n\nQuestion: What is the active route and budget for Node-700?\nAnswer concisely:",
            max_tokens=32,
        )
        ev_b_answered1.set()

        # Step 6, 7, 8: Receive incremental update, verify obsolete data gone, checkout seq=4 history
        assert ev_msg2_sent.wait(timeout=30.0)
        res2 = cons.poll_and_apply_ring()
        assert res2 is not None
        view_now = cons.adapter.active_view()
        ctx2 = cons.render_context("r3")
        ans2 = _llama_query(
            model_b,
            f"Context:\n{ctx2}\n\nQuestion: What is the active route and budget for Node-700?\nAnswer concisely:",
            max_tokens=32,
        )
        hist_at_4 = cons.checkout_history(target_seq=4)
        out_q.put({
            "role": "B_initial",
            "ans1": ans1,
            "ans2": ans2,
            "active_now": {k: v["value"] for k, v in view_now.items()},
            "hist_at_4": {k: v["value"] for k, v in hist_at_4.items()},
        })
    finally:
        ring.close()


def _process_b_restarted(arena_shm: str, ident1: Any, ident2: Any, out_q: Any) -> None:
    arena = VectLinkArenaTransport.open(arena_shm)
    try:
        cons = CMCPAdapter("multiagent-session")
        for ident in (ident1, ident2):
            raw = arena.read_block(ident)
            assert raw is not None
            cons.decode_and_apply(raw)
        out_q.put({"role": "B_restarted", "recovered_active": {k: v["value"] for k, v in cons.active_view().items()}})
    finally:
        arena.close()


def run_demo() -> Dict[str, Any]:
    model_a = os.environ.get("CMCP_MODEL_A", r"C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf")
    model_b = os.environ.get("CMCP_MODEL_B", r"C:\Users\sgi.lm\Desktop\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    ring_shm = f"/vcmcp_ma_ring_{uuid.uuid4().hex[:8]}"
    arena_shm = f"/vcmcp_ma_arena_{uuid.uuid4().hex[:8]}"

    ctx = multiprocessing.get_context("spawn")
    ev_b_ready = ctx.Event()
    ev_msg1_sent = ctx.Event()
    ev_b_answered1 = ctx.Event()
    ev_msg2_sent = ctx.Event()
    ev_done = ctx.Event()
    q = ctx.Queue()

    pa = ctx.Process(target=_process_a_researcher, args=(ring_shm, arena_shm, model_a, ev_b_ready, ev_msg1_sent, ev_b_answered1, ev_msg2_sent, ev_done, q))
    pb = ctx.Process(target=_process_b_executor, args=(ring_shm, arena_shm, model_b, ev_b_ready, ev_msg1_sent, ev_b_answered1, ev_msg2_sent, q))

    pa.start()
    pb.start()
    pb.join(timeout=60.0)
    assert pb.exitcode == 0

    items = [q.get(timeout=10.0), q.get(timeout=10.0)]
    by_role = {x["role"]: x for x in items}

    # Step 9: Spawn a fresh restarted Consumer B process recovering from KVArena
    pb2 = ctx.Process(target=_process_b_restarted, args=(arena_shm, by_role["A"]["ident1"], by_role["A"]["ident2"], q))
    pb2.start()
    pb2.join(timeout=20.0)
    assert pb2.exitcode == 0
    by_role["B_restarted"] = q.get(timeout=10.0)

    ev_done.set()
    pa.join(timeout=20.0)
    assert pa.exitcode == 0
    return by_role


if __name__ == "__main__":
    res = run_demo()
    # Strip non-JSON KVIdentity objects for clean display
    res["A"] = {"private_summary": res["A"]["private_summary"]}
    print(json.dumps(res, indent=2))
