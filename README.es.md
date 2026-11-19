# VectLink-CMCP

Arquitectura experimental de memoria compartida en dos niveles que combina **CMCP** (Nivel 1: estado semántico) y **Vect-Link** (Nivel 2: transporte SHM y arena KV Copy-on-Write) para sistemas multiagente con LLMs.

> **Veredicto Científico Honesto:** Ver [FINAL_INTEGRATION_REPORT.md](FINAL_INTEGRATION_REPORT.md).
> - **Integración Nivel 1 + Nivel 2:** Funciona limpiamente (`16/16` tests Python, `ctest` Release + ASan/UBSan, demo real en 2 procesos OS).
> - **Vect-Link SHM para deltas CMCP de `~350 B`:** **Sin ventaja E2E apreciable** (`-0.056 ms` en IPC, `-2.06 ms` sobre `3,245 ms` E2E frente a socket TCP loopback). Para traspasos semánticos pequeños alcanza IPC convencional.
> - **Vect-Link `KVArena` para reutilización KV en el mismo modelo:** **Demostrado** en `Qwen2.5-0.5B` (`0.85 ms` para restaurar `1.29 MB` de estado KV, **-71.0% prefill**, **-22.3% latencia E2E**). La reutilización KV entre arquitecturas distintas se rechaza estrictamente.

## Estructura
- `src/vectlink_cmcp/`: `cmcp_adapter.py`, `vectlink_adapter.py`, `integration_runtime.py`, `kv_reuse_bridge.py`
- `csrc/vl_cmcp_kv_bridge.c`: Puente C11 `llama.cpp` + `vectlink.dll` en dos procesos para reutilización KV
- `tests/`: Fase 2 (13 tests de transferencia) y Fase 3 (3 tests de compatibilidad KV) + `tests/c/test_vl_cmcp_c_api.c`
- `examples/two_process_multiagent.py`: Demo real multiagente en 2 procesos del SO
- `benchmarks/`: `run_phase3_kv_reuse.py`, `run_phase4_comparative.py`
- `docs/`: [INTEGRATION_COMPATIBILITY_AUDIT.md](docs/audit/INTEGRATION_COMPATIBILITY_AUDIT.md), [INTEGRATED_MEMORY_ARCHITECTURE.md](docs/architecture/INTEGRATED_MEMORY_ARCHITECTURE.md), [INTEGRATION_BENCHMARK_METHODOLOGY.md](docs/benchmarks/INTEGRATION_BENCHMARK_METHODOLOGY.md), [INTEGRATION_AUDIT_REPORT.md](docs/audit/INTEGRATION_AUDIT_REPORT.md)

## Licencia y Atribución
Apache License 2.0 — ver [LICENSE](LICENSE) y [NOTICE](NOTICE).
