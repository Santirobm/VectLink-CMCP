# Integration Benchmark Methodology

## 1. Experimental Separation

- **Phase 2 (Context Transfer Correctness):** 13 test functions covering all 15 mandatory edge cases (initial state, incremental update, duplicate idempotency, out-of-order rejection, invalidation, tombstones, fact correction, `checkout_seq` history, consumer restart, producer restart, corrupt payload, truncated payload, incompatible schema, cross-process SHM reconnection, 12 concurrent readers on `KVArena`).
- **Phase 3 (Same-Model KV Reuse vs Cross-Arch Rejection):** Two real OS processes (`vl_cmcp_kv_bridge.exe`) on `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` (`reps = 3`) comparing `A_cold_full_history`, `B_vectlink_kv_full`, `C_cmcp_prefill`, and `D_cmcp_kv_reuse`, plus explicit cross-arch rejection (`Qwen2.5-0.5B -> Llama-3.2-1B`).
- **Phase 4 (Comparative Benchmark):**
  1. **IPC Microbenchmark:** 1,000 CMCP `R3` delta envelopes × 3 reps comparing `ConventionalSocketTransport` (loopback TCP, `TCP_NODELAY`) vs `VectLinkRingTransport` (SHM).
  2. **End-to-End LLM Benchmark:** `n = 50` held-out `eval` scenarios (5 disjoint domains, Turn 8 composite query, `reps = 3`, greedy decoding, `max_tokens = 64`) on `Llama-3.2-1B-Instruct-Q4_K_M.gguf` comparing `full_history_socket`, `cmcp_inc_socket`, and `cmcp_inc_vectlink`.

## 2. Statistical Reporting

- Scenario-clustered bootstrap (`5,000` resamples, `seed = 42`) over the `n = 50` scenarios (never treating 3 repetitions of the same scenario as independent samples).
- Latency metrics report `mean`, `95% CI`, `P50` (median), `P95`, and `std`.
- Primary decision metric: complete end-to-end latency (`e2e_accum_ms = extract + ipc + tokenize + prefill + decode`) alongside quality (`exact_match`, `relation_recovery`, `constraint_satisfaction`).
