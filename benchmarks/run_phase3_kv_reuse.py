"""Phase 3 Real Two-Process KV Reuse Experiment (Section 5.2).

Compares across 4 conditions on Qwen2.5-0.5B-Instruct-Q4_K_M.gguf (reps=3):
  A. cold_full_history: Fresh process prefilling full history + query
  B. vectlink_kv_full: Fresh process restoring full-history KV snapshot from Vect-Link KVArena
  C. cmcp_prefill: Fresh process prefilling CMCP R3 context + query
  D. cmcp_kv_reuse: Fresh process restoring CMCP R3 KV snapshot from Vect-Link KVArena

Also verifies that cross-architecture KV restore (Qwen2.5-0.5B -> Llama-3.2-1B)
is strictly rejected by Vect-Link vtlk_kv_restore().
"""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vectlink_cmcp import CMCPAdapter, StateOp  # noqa: E402


def _bridge_env() -> Dict[str, str]:
    env = os.environ.copy()
    extra = r"C:\msys64\clang64\bin;C:\Users\sgi.lm\Desktop\Vect-Link\build;C:\Users\sgi.lm\Desktop\llama.cpp\build\bin"
    env["PATH"] = extra + ";" + env.get("PATH", "")
    return env


def _bridge_exe() -> str:
    return str(ROOT / "build" / "vl_cmcp_kv_bridge.exe")


def _run_one_role(args: List[str]) -> Dict[str, Any]:
    out = subprocess.check_output([_bridge_exe(), *args], env=_bridge_env(), text=True, encoding="utf-8")
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
    model_qwen = os.environ.get("CMCP_MODEL_A", r"C:\Users\sgi.lm\Desktop\models\qwen2.5-0.5b-instruct-q4_k_m.gguf")
    model_llama = os.environ.get("CMCP_MODEL_B", r"C:\Users\sgi.lm\Desktop\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf")

    full_hist = (
        "History:\n"
        "[T1] Primary node Node-100 is active; anchor=ANC-100; budget<=90ms; route=R-100.\n"
        "[T2] Node-100 connects via Relay-200 to Sink-300. Ignore standby Node-100B.\n"
        "[T3] Tighten budget<=40ms; update route=R-200; decision: tier_mode=strict_v2.\n"
    )
    adp = CMCPAdapter("kv-exp")
    adp.apply_local_ops([
        StateOp("e1", "UPSERT", "entity", "Node-100", "primary; anchor=ANC-100", seq=1, version=1),
        StateOp("e2", "UPSERT", "entity", "Node-100B", "standby_ignore", seq=2, version=1),
        StateOp("rel", "UPSERT", "relation", "path", "Node-100 -> Relay-200 -> Sink-300", seq=3, version=1),
        StateOp("c1", "UPSERT", "constraint", "budget", "budget<=40ms", seq=4, version=2),
        StateOp("f1", "UPSERT", "fact", "route", "R-200", seq=5, version=2),
        StateOp("d1", "UPSERT", "decision", "tier_mode", "strict_v2", seq=6, version=2),
    ])
    cmcp_ctx = "Context:\n" + adp.render("r3") + "\n"
    query = "\nQuestion: Report the active path, active route, and active budget for Node-100.\nAnswer in plain text, one concise sentence:"

    res_dir = ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        p_full = Path(tmp) / "full.txt"
        p_cmcp = Path(tmp) / "cmcp.txt"
        p_q = Path(tmp) / "query.txt"
        p_full.write_text(full_hist, encoding="utf-8")
        p_cmcp.write_text(cmcp_ctx, encoding="utf-8")
        p_q.write_text(query, encoding="utf-8")

        # 1. Verify cross-architecture rejection in real C bridge
        shm_rej = f"/vcmcp_kv_rej_{uuid.uuid4().hex[:8]}"
        prod_rej = subprocess.Popen(
            [_bridge_exe(), "--role", "producer", "--model", model_qwen, "--shm", shm_rej, "--prefix-file", str(p_cmcp)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=_bridge_env(),
        )
        try:
            ready_line = prod_rej.stdout.readline()  # type: ignore[union-attr]
            assert '"ready":true' in ready_line
            rej_proc = subprocess.run(
                [_bridge_exe(), "--role", "consumer", "--model", model_llama, "--shm", shm_rej, "--snapshot-id", "1", "--query-file", str(p_q)],
                capture_output=True, text=True, env=_bridge_env(),
            )
            cross_arch_rejected = rej_proc.returncode != 0
            cross_arch_stderr = rej_proc.stderr.strip()
        finally:
            if prod_rej.stdin:
                prod_rej.stdin.close()
            prod_rej.wait(timeout=10)

        # 2. Run 3 reps of A, B, C, D on Qwen2.5-0.5B
        for rep in range(1, 4):
            # A: cold_full_history
            a_res = _run_one_role(["--role", "cold", "--model", model_qwen, "--prefix-file", str(p_full), "--query-file", str(p_q), "--max-tokens", "32"])
            rows.append({"rep": rep, "condition": "A_cold_full_history", **a_res})

            # B: vectlink_kv_full (two independent OS processes)
            shm_b = f"/vcmcp_kv_b_{uuid.uuid4().hex[:8]}"
            pb = subprocess.Popen(
                [_bridge_exe(), "--role", "producer", "--model", model_qwen, "--shm", shm_b, "--prefix-file", str(p_full)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=_bridge_env(),
            )
            try:
                _ = pb.stdout.readline()  # type: ignore[union-attr]
                b_res = _run_one_role(["--role", "consumer", "--model", model_qwen, "--shm", shm_b, "--snapshot-id", "1", "--query-file", str(p_q), "--max-tokens", "32"])
                rows.append({"rep": rep, "condition": "B_vectlink_kv_full", **b_res})
            finally:
                if pb.stdin:
                    pb.stdin.close()
                pb.wait(timeout=10)

            # C: cmcp_prefill
            c_res = _run_one_role(["--role", "cold", "--model", model_qwen, "--prefix-file", str(p_cmcp), "--query-file", str(p_q), "--max-tokens", "32"])
            rows.append({"rep": rep, "condition": "C_cmcp_prefill", **c_res})

            # D: cmcp_kv_reuse (two independent OS processes)
            shm_d = f"/vcmcp_kv_d_{uuid.uuid4().hex[:8]}"
            pd = subprocess.Popen(
                [_bridge_exe(), "--role", "producer", "--model", model_qwen, "--shm", shm_d, "--prefix-file", str(p_cmcp)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=_bridge_env(),
            )
            try:
                _ = pd.stdout.readline()  # type: ignore[union-attr]
                d_res = _run_one_role(["--role", "consumer", "--model", model_qwen, "--shm", shm_d, "--snapshot-id", "1", "--query-file", str(p_q), "--max-tokens", "32"])
                rows.append({"rep": rep, "condition": "D_cmcp_kv_reuse", **d_res})
            finally:
                if pd.stdin:
                    pd.stdin.close()
                pd.wait(timeout=10)

    csv_path = res_dir / "phase3_kv_reuse_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary: Dict[str, Any] = {
        "model": Path(model_qwen).name,
        "cross_arch_rejection": {"rejected": cross_arch_rejected, "stderr": cross_arch_stderr},
        "conditions": {},
    }
    for cond in ("A_cold_full_history", "B_vectlink_kv_full", "C_cmcp_prefill", "D_cmcp_kv_reuse"):
        crows = [r for r in rows if r["condition"] == cond]
        summary["conditions"][cond] = {
            k: round(statistics.mean(float(r[k]) for r in crows), 3)
            for k in ("init_ms", "restore_ms", "prefill_ms", "ttft_ms", "decode_ms", "e2e_ms", "rss_mib", "kv_bytes")
        }
        summary["conditions"][cond]["output_sample"] = crows[0]["output_text"][:100]

    (res_dir / "phase3_kv_reuse_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
