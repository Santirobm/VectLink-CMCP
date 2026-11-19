# Integration Security, Concurrency & Robustness Audit

## 1. Test & Sanitizer Verification

| Suite | Scope | Result |
|---|---|---|
| `ctest --test-dir build` (Release) | `test_vl_cmcp_c_api` (C11 SHM ring + CRC32C) | **1/1 PASSED** (`0.04s`) |
| `ctest --test-dir build-san` (ASan + UBSan) | `test_vl_cmcp_c_api` under `-fsanitize=address,undefined` | **1/1 PASSED** (`0.12s`, 0 leaks/UB) |
| `pytest tests/ -v` | 13 Phase 2 context-transfer tests + 3 Phase 3 KV-compat tests | **16/16 PASSED** |
| `examples/two_process_multiagent.py` | 2 independent OS processes (`Qwen2.5-0.5B` + `Llama-3.2-1B`) | **PASSED** (all 9 steps) |

## 2. Robustness & Security Findings

1. **Race Conditions & SPSC Discipline:** `VectLinkRing` is strictly Single-Producer, Single-Consumer. `VectLinkRingTransport` enforces one writer and one reader per ring instance; multi-reader fan-out uses immutable sealed blocks in `KVArena` (`test_15_concurrent_readers_on_kv_arena`, 12 concurrent readers across 6 threads).
2. **Partial Reads / Corrupt / Truncated Payloads:**
   - `VCM1` envelope validates magic (`VCM1`), `schema_version == 1`, exact `payload_len`, and `crc32c` before passing bytes to CMCP TLV `decode_frame()` (which also checks TLV CRC32C).
   - Verified by `test_11_corrupt_payload_rejected`, `test_12_truncated_payload_rejected`, and `test_13_incompatible_schema_version_rejected`.
3. **Version Confusion & Tombstones:**
   - `CMCPStateStore` enforces monotonic `seq` per `fact_id`, idempotent duplicate suppression (`DUPLICATE_IGNORED`), stale rejection (`STALE_IGNORED`), and tombstone resurrection blocking (`test_03`–`test_08`).
4. **KV Snapshot Isolation & Cross-Arch Rejection:**
   - `vtlk_kv_restore()` in `vl_cmcp_kv_bridge.c` and `assert_kv_compatible()` in `kv_reuse_bridge.py` reject mismatched model weights, architecture, tokenizer, RoPE, or dtype (`Qwen2.5-0.5B -> Llama-3.2-1B` rejected with exit code 11).
5. **Windows SHM Lifetime Constraint (Documented Limitation):**
   - On Windows (`CreateFileMapping`), kernel SHM objects vanish when the last open handle closes. Multi-process restarts require either an active producer/coordinator handle or `l3_dir` persistence (which is available in `Vect-Link` C API `vtlk_kv_arena_create_ex` but not yet exposed in Python `KVArena.create`).
6. **Note on TSan:** TSan is not supported on Windows Clang/MinGW; multi-reader concurrency was tested via `ThreadPoolExecutor` + `multiprocessing` spawn processes, and memory safety was verified via ASan + UBSan.
