"""Phase 4 Comparative Benchmark (Section 6).

1. Pure IPC Microbenchmark (1,000 CMCP R3 delta packets x 3 reps):
   - ConventionalSocketTransport (loopback TCP) vs VectLinkRingTransport (SHM)
2. End-to-End Multi-Turn Comparative Benchmark (n=50 eval scenarios x 3 reps, Turn 8):
   - full_history_socket
   - cmcp_inc_socket
   - cmcp_inc_vectlink
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

CMCP_ROOT = Path(os.environ.get("CMCP_ROOT", ROOT.parent / "Cross-Model-Context-Protocol"))
if str(CMCP_ROOT) not in sys.path:
    sys.path.insert(0, str(CMCP_ROOT))

from experiments.cmcp22.dataset import dataset_sha256, get_scenarios  # noqa: E402
from experiments.cmcp22.extractors import DeterministicExtractor  # noqa: E402
from experiments.cmcp22.run_multiturn_benchmark import (  # noqa: E402
    PersistentLlamaRunner,
    _bootstrap_ci,
    _score_query,
)
from vectlink_cmcp import (  # noqa: E402
    CMCPAdapter,
    ConsumerSession,
    ConventionalSocketTransport,
    ProducerSession,
    StateOp,
    VectLinkRingTransport,
)


def run_ipc_microbenchmark(n_msgs: int = 1000, reps: int = 3) -> Dict[str, Any]:
    adp = CMCPAdapter("ipc-bench")
    ops = [
        StateOp("e1", "UPSERT", "entity", "Node-100", "primary", seq=1, version=1),
        StateOp("rel", "UPSERT", "relation", "path", "Node-100 -> Relay-2 -> Sink-3", seq=2, version=1),
        StateOp("c1", "UPSERT", "constraint", "budget", "budget<=40ms", seq=3, version=1),
    ]
    pkt = adp.encode_ops(ops, seq=3)
    pkt_len = len(pkt)

    results: Dict[str, Any] = {"payload_bytes": pkt_len, "n_msgs": n_msgs, "reps": reps, "transports": {}}

    # 1. Conventional Loopback TCP Socket
    sock_pub_us, sock_con_us, sock_tot_ms = [], [], []
    for _ in range(reps):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.connect(("127.0.0.1", port))
        conn, _ = srv.accept()
        srv.close()
        cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        t_prod = ConventionalSocketTransport(cli)
        t_cons = ConventionalSocketTransport(conn)
        cons_times: List[float] = []

        def _reader() -> None:
            for _i in range(n_msgs):
                tc0 = time.perf_counter()
                _ = t_cons.consume_blocking()
                cons_times.append((time.perf_counter() - tc0) * 1e6)

        th = threading.Thread(target=_reader, daemon=True)
        th.start()
        t_all0 = time.perf_counter()
        for i in range(n_msgs):
            tp0 = time.perf_counter()
            t_prod.publish(pkt, sequence=i + 1)
            sock_pub_us.append((time.perf_counter() - tp0) * 1e6)
        th.join(timeout=30.0)
        sock_tot_ms.append((time.perf_counter() - t_all0) * 1000.0)
        sock_con_us.extend(cons_times)
        t_prod.close()
        t_cons.close()

    results["transports"]["conventional_socket"] = {
        "publish_us_mean": round(statistics.mean(sock_pub_us), 2),
        "consume_us_mean": round(statistics.mean(sock_con_us), 2),
        "total_ms_mean": round(statistics.mean(sock_tot_ms), 3),
        "throughput_mb_s": round((n_msgs * pkt_len / (1024 * 1024)) / (statistics.mean(sock_tot_ms) / 1000.0), 2),
    }

    # 2. Vect-Link SHM Ring
    vl_pub_us, vl_con_us, vl_tot_ms = [], [], []
    for _ in range(reps):
        shm = f"/vcmcp_ipc_{uuid.uuid4().hex[:8]}"
        r_prod = VectLinkRingTransport.create(shm, capacity=64, max_payload_bytes=4096)
        r_cons = VectLinkRingTransport.open(shm)
        t_all0 = time.perf_counter()
        for i in range(n_msgs):
            tp0 = time.perf_counter()
            r_prod.publish(pkt, sequence=i + 1)
            vl_pub_us.append((time.perf_counter() - tp0) * 1e6)
            tc0 = time.perf_counter()
            rec = r_cons.try_consume()
            assert rec is not None
            vl_con_us.append((time.perf_counter() - tc0) * 1e6)
        vl_tot_ms.append((time.perf_counter() - t_all0) * 1000.0)
        r_cons.close()
        r_prod.close(unlink=True)

    results["transports"]["vectlink_ring"] = {
        "publish_us_mean": round(statistics.mean(vl_pub_us), 2),
        "consume_us_mean": round(statistics.mean(vl_con_us), 2),
        "total_ms_mean": round(statistics.mean(vl_tot_ms), 3),
        "throughput_mb_s": round((n_msgs * pkt_len / (1024 * 1024)) / (statistics.mean(vl_tot_ms) / 1000.0), 2),
    }
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["dev", "tune", "eval"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model-b", default=os.environ.get("CMCP_MODEL_B", r"C:\Users\sgi.lm\Desktop\models\Llama-3.2-1B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--tag", default="phase4_eval")
    args = ap.parse_args()

    res_dir = ROOT / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    ipc_micro = run_ipc_microbenchmark(n_msgs=1000, reps=3)
    (res_dir / f"phase4_ipc_micro_{args.tag}.json").write_text(json.dumps(ipc_micro, indent=2), encoding="utf-8")

    scenarios = get_scenarios(args.split)
    methods = ["full_history_socket", "cmcp_inc_socket", "cmcp_inc_vectlink"]
    det_ext = DeterministicExtractor()

    # Persistent loopback socket pair for socket IPC baseline
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cli.connect(("127.0.0.1", srv.getsockname()[1]))
    conn, _ = srv.accept()
    srv.close()
    cli.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock_prod = ConventionalSocketTransport(cli)
    sock_cons = ConventionalSocketTransport(conn)

    shm_name = f"/vcmcp_p4_{uuid.uuid4().hex[:8]}"
    vl_prod = VectLinkRingTransport.create(shm_name, capacity=16, max_payload_bytes=16384)
    vl_cons = VectLinkRingTransport.open(shm_name)
    runner_b = PersistentLlamaRunner(args.model_b, n_ctx=1024, threads=4)

    csv_path = res_dir / f"phase4_runs_{args.tag}.csv"
    fieldnames = [
        "scenario_id", "domain", "method", "rep",
        "wire_bytes", "publish_ms", "recover_ms", "ipc_ms",
        "input_tokens", "output_tokens", "prefill_ms", "decode_ms", "e2e_accum_ms", "rss_mib",
        "exact_match", "f1", "entity_recall", "relation_recovery",
        "constraint_satisfaction", "temporal_consistency", "contradiction_resolution",
        "task_completion_rate", "obsolete_info_rate",
    ]
    rows: List[Dict[str, Any]] = []
    try:
        _ = runner_b.query("Warmup: OK", max_tokens=2)
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for rep in range(1, args.reps + 1):
                for sc in scenarios:
                    qspec = sc.eval_queries[3]  # Turn 8 composite query
                    gold_str = " ".join(qspec["expected_keywords"])
                    hist_lines: List[str] = []
                    all_ops: List[StateOp] = []
                    ext_ms = 0.0
                    seq = 0
                    for turn in sc.turns:
                        if turn.speaker == "agent_a":
                            hist_lines.append(turn.utterance)
                            ops, ms, _ = det_ext.extract_turn(turn.utterance, seq)
                            ext_ms += ms
                            for op in ops:
                                seq = max(seq, op.seq)
                            all_ops.extend(ops)
                    delta_ops = all_ops[-3:]
                    base_ops = all_ops[:-3]

                    for method in methods:
                        if method == "full_history_socket":
                            raw = "\n".join(hist_lines).encode("utf-8")
                            tp0 = time.perf_counter()
                            sock_prod.publish(raw, sequence=rep)
                            pub_ms = (time.perf_counter() - tp0) * 1000.0
                            tr0 = time.perf_counter()
                            got = sock_cons.consume_blocking()
                            rec_ms = (time.perf_counter() - tr0) * 1000.0
                            wire_bytes = len(raw)
                            prompt = f"History:\n{got.decode('utf-8')}\n\nQuestion: {qspec['query']}\nAnswer in plain text, one concise sentence:"
                            cur_ext_ms = 0.0
                        else:
                            p_adp = CMCPAdapter(sc.scenario_id)
                            p_adp.apply_local_ops(all_ops)
                            c_adp = CMCPAdapter(sc.scenario_id)
                            c_adp.apply_local_ops(base_ops)

                            if method == "cmcp_inc_socket":
                                prod_s = ProducerSession(p_adp, sock_prod)
                                cons_s = ConsumerSession(c_adp, sock_cons)
                                tp0 = time.perf_counter()
                                pkt = prod_s.commit_and_publish(delta_ops)
                                pub_ms = (time.perf_counter() - tp0) * 1000.0
                                tr0 = time.perf_counter()
                                cons_s.consume_socket_and_apply()
                                ctx_str = cons_s.render_context("r3")
                                rec_ms = (time.perf_counter() - tr0) * 1000.0
                            else:
                                prod_s = ProducerSession(p_adp, vl_prod)
                                cons_s = ConsumerSession(c_adp, vl_cons)
                                tp0 = time.perf_counter()
                                pkt = prod_s.commit_and_publish(delta_ops)
                                pub_ms = (time.perf_counter() - tp0) * 1000.0
                                tr0 = time.perf_counter()
                                res = cons_s.poll_and_apply_ring()
                                assert res is not None
                                ctx_str = cons_s.render_context("r3")
                                rec_ms = (time.perf_counter() - tr0) * 1000.0

                            wire_bytes = len(pkt)
                            prompt = f"Context:\n{ctx_str}\n\nQuestion: {qspec['query']}\nAnswer in plain text, one concise sentence:"
                            cur_ext_ms = ext_ms

                        resp = runner_b.query(prompt, max_tokens=args.max_tokens)
                        out_txt = resp.get("output_text", "")
                        sc_m = _score_query(out_txt, qspec, gold_str)
                        tok_ms = float(resp.get("tokenize_ms", 0.0))
                        pre_ms = float(resp.get("prefill_ms", 0.0))
                        dec_ms = float(resp.get("eval_ms", 0.0))
                        ipc_ms = pub_ms + rec_ms
                        e2e_ms = cur_ext_ms + ipc_ms + tok_ms + pre_ms + dec_ms

                        row = {
                            "scenario_id": sc.scenario_id,
                            "domain": sc.domain,
                            "method": method,
                            "rep": rep,
                            "wire_bytes": wire_bytes,
                            "publish_ms": round(pub_ms, 4),
                            "recover_ms": round(rec_ms, 4),
                            "ipc_ms": round(ipc_ms, 4),
                            "input_tokens": int(resp.get("n_input_tokens", 0)),
                            "output_tokens": int(resp.get("n_gen_tokens", 0)),
                            "prefill_ms": round(pre_ms, 3),
                            "decode_ms": round(dec_ms, 3),
                            "e2e_accum_ms": round(e2e_ms, 3),
                            "rss_mib": round(float(resp.get("rss_mib", 0.0)), 2),
                            "exact_match": round(sc_m["exact_match"], 4),
                            "f1": round(sc_m["f1"], 4),
                            "entity_recall": round(sc_m["entity_recall"], 4),
                            "relation_recovery": round(sc_m["relation_recovery"], 4),
                            "constraint_satisfaction": round(sc_m["constraint_satisfaction"], 4),
                            "temporal_consistency": round(sc_m["temporal_consistency"], 4),
                            "contradiction_resolution": round(sc_m["contradiction_resolution"], 4),
                            "task_completion_rate": round(sc_m["task_completion_rate"], 4),
                            "obsolete_info_rate": round(sc_m["obsolete_info_rate"], 4),
                        }
                        writer.writerow(row)
                        fcsv.flush()
                        rows.append(row)
    finally:
        runner_b.close()
        vl_cons.close()
        vl_prod.close(unlink=True)
        sock_prod.close()
        sock_cons.close()

    stat_keys = [
        "wire_bytes", "ipc_ms", "input_tokens", "output_tokens",
        "prefill_ms", "decode_ms", "e2e_accum_ms", "rss_mib",
        "exact_match", "f1", "entity_recall", "relation_recovery",
        "constraint_satisfaction", "temporal_consistency", "contradiction_resolution",
        "task_completion_rate", "obsolete_info_rate",
    ]
    sc_ids = [s.scenario_id for s in scenarios]
    sc_means: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in stat_keys} for m in methods}
    summary: Dict[str, Any] = {
        "config": {
            "split": args.split,
            "num_scenarios": len(scenarios),
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "model_b_name": Path(args.model_b).name,
            "dataset_sha256": dataset_sha256(),
            "bootstrap_seed": 42,
            "tag": args.tag,
        },
        "ipc_microbenchmark": ipc_micro,
        "methods": {},
        "paired_vs_full_history": {},
    }

    for m in methods:
        m_rows = [r for r in rows if r["method"] == m]
        for sid in sc_ids:
            srows = [r for r in m_rows if r["scenario_id"] == sid]
            for k in stat_keys:
                sc_means[m][k].append(float(statistics.mean([float(r[k]) for r in srows])))
        m_sum: Dict[str, Any] = {"n_scenarios": len(sc_ids), "n_rows": len(m_rows)}
        for k in stat_keys:
            mu, lo, hi = _bootstrap_ci(sc_means[m][k])
            entry: Dict[str, Any] = {"mean": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            if k in ("ipc_ms", "prefill_ms", "decode_ms", "e2e_accum_ms"):
                raw_vals = [float(r[k]) for r in m_rows]
                entry["p50"] = round(float(np.percentile(raw_vals, 50)), 4)
                entry["p95"] = round(float(np.percentile(raw_vals, 95)), 4)
                entry["std"] = round(float(np.std(raw_vals, ddof=1)) if len(raw_vals) > 1 else 0.0, 4)
            m_sum[k] = entry
        summary["methods"][m] = m_sum

    for m in methods:
        if m == "full_history_socket":
            continue
        p_diff: Dict[str, Any] = {}
        for k in ["ipc_ms", "input_tokens", "prefill_ms", "e2e_accum_ms", "exact_match", "relation_recovery", "constraint_satisfaction"]:
            diffs = [a - b for a, b in zip(sc_means[m][k], sc_means["full_history_socket"][k])]
            mu, lo, hi = _bootstrap_ci(diffs)
            p_diff[k] = {"mean_diff": round(mu, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        summary["paired_vs_full_history"][m] = p_diff

    (res_dir / f"phase4_summary_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
