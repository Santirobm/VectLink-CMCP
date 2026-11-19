# VectLink-CMCP

Experimental two-level shared-memory architecture combining **CMCP** (Level 1 semantic state) and **Vect-Link** (Level 2 SHM ring transport & CoW KV-cache arena) for multi-agent LLM systems.

> **Honest Research Verdict:** See [FINAL_INTEGRATION_REPORT.md](FINAL_INTEGRATION_REPORT.md).
> - **Level 1 + Level 2 Integration:** Works cleanly (`16/16` Python tests, `ctest` Release + ASan/UBSan, real 2-process demo).
> - **Vect-Link SHM for `~350 B` CMCP deltas:** **No meaningful E2E gain** (`-0.056 ms` IPC, `-2.06 ms` out of `3,245 ms` E2E vs loopback TCP socket). Conventional IPC suffices for pure semantic text handoffs.
> - **Vect-Link `KVArena` for Same-Model KV Reuse:** **Demonstrated** on `Qwen2.5-0.5B` (`0.85 ms` restore of `1.29 MB` KV state, **-71.0% prefill**, **-22.3% E2E latency**). Cross-architecture KV reuse is strictly rejected.

## Structure
- `src/vectlink_cmcp/`: `cmcp_adapter.py`, `vectlink_adapter.py`, `integration_runtime.py`, `kv_reuse_bridge.py`
- `csrc/vl_cmcp_kv_bridge.c`: C11 `llama.cpp` + `vectlink.dll` two-process KV-reuse bridge
- `tests/`: Phase 2 (13 context-transfer tests) & Phase 3 (3 KV-compat tests) + `tests/c/test_vl_cmcp_c_api.c`
- `examples/two_process_multiagent.py`: Real 2-process OS multi-agent demo
- `benchmarks/`: `run_phase3_kv_reuse.py`, `run_phase4_comparative.py`
- `docs/`: [INTEGRATION_COMPATIBILITY_AUDIT.md](docs/audit/INTEGRATION_COMPATIBILITY_AUDIT.md), [INTEGRATED_MEMORY_ARCHITECTURE.md](docs/architecture/INTEGRATED_MEMORY_ARCHITECTURE.md), [INTEGRATION_BENCHMARK_METHODOLOGY.md](docs/benchmarks/INTEGRATION_BENCHMARK_METHODOLOGY.md), [INTEGRATION_AUDIT_REPORT.md](docs/audit/INTEGRATION_AUDIT_REPORT.md)

## Quick Reproduction
```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
pytest tests/ -v
python examples/two_process_multiagent.py
```

## License & Attribution
Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
