# Integration Compatibility Audit: Vect-Link + CMCP

**Date:** 2026-11-18
**Target Repository:** `VectLink-CMCP` (branch `main`, initial commit pending)
**Reference Repositories (Untouched):**
- `Vect-Link` (`main` @ `79ae1a9`) — Apache-2.0
- `Cross-Model-Context-Protocol` (`main` @ `0a7b649`) — Apache-2.0

---

## 1. Repository Audit Summary

### 1.1 `VectLink-CMCP`
- Initialized as an empty Git repository (`main`, no prior commits, no code/tests/CI/LICENSE yet).
- We will establish Apache-2.0 licensing with full attribution to both upstream projects.

### 1.2 `Vect-Link` (`79ae1a9`)
Audited public headers (`include/vectlink/*.h`) and Python bindings (`python/vectlink/{__init__,ring,kv,_ffi}.py`):
- **Shared-Memory Ring Transport (`VectLinkRing`, `WriteSlot`, `ReadSlot`):**
  - `vtlk_ring_init` / `vtlk_ring_attach` over `vtlk_shm_create` / `vtlk_shm_open`.
  - Zero-copy slot loan API (`acquire_write` / `WriteSlot.write_bytes` / `commit` and `acquire_read` / `ReadSlot.buffer` / `release`) with `QuantScheme.RAW` for arbitrary binary payloads.
  - 64-byte header + payload hardware-accelerated CRC32C validation (`CrcMismatchError`).
  - **Concurrency limit:** Strictly **SPSC** (Single-Producer, Single-Consumer). Multiple concurrent producers or consumers on the same ring are unsafe without external synchronization or per-consumer rings.
- **Copy-on-Write KV Arena (`KVArena`, `KVIdentity`, `Branch`, `Snapshot`):**
  - Shared-memory paged block arena (`create`, `open`, `alloc_block`, `open_block`, `seal_block`, `lookup_block`, `new_branch`, `create_snapshot`, `fork`, `restore`).
  - `KVIdentity` validates `token_prefix_hash`, `model_arch_hash`, `rope_config_hash`, `lora_adapter_hash`, `pos_offset`, `token_count`, `kv_dtype`, `layer_count`, `head_dim`, `kv_head_count`.
  - Supports **multi-reader concurrency** on sealed immutable blocks/snapshots and optional `l3_dir` disk persistence.
  - **Scope boundary:** `KVArena` stores raw opaque KV state bytes (e.g. `llama_state_seq_get_data` / `llama_state_seq_set_data` in `llama.cpp`). It does **not** perform cross-architecture tensor translation.

### 1.3 `Cross-Model-Context-Protocol` (`0a7b649`)
Audited `representations/state_store.py`, `representations/binary/`, and `FINAL_RESEARCH_REPORT_V3.md` / `V4.md`:
- **`CMCPStateStore` & `StateOp`:**
  - Versioned `UPSERT` / `INVALIDATE` operations with `(fact_id, seq, version, agent_id, digest)`.
  - Idempotent duplicate handling, out-of-order stale rejection, tombstone resurrection prevention, `lww` / `reject` conflict policies, `save_snapshot`, `resolve_reference`, `checkout_seq(target_seq)`, and `compact()`.
  - Semantic rendering: `render_for_llm(style="r1"|"r2"|"r3")` (`r3` XML-lite `<relations><entities><constraints><decisions><facts>`).
- **Binary TLV Framing (`encode_frame` / `decode_frame`):**
  - Compact zlib-compressed typed TLV frames with CRC32C integrity (`~250 B` incremental delta vs `~686 B` full history).
- **Verified Empirical Findings (CMCP 2.2 / 2.2b):**
  - Homogeneous (`Qwen2.5-0.5B -> Qwen2.5-0.5B`): `+0.22 EM`, `-101.3 ms` E2E speedup.
  - Heterogeneous (`Qwen2.5-0.5B -> Llama-3.2-1B`): `-0.01 EM` with `R3`, `-257.6 ms` prefill, but `+442.1 ms` E2E due to decode verbosity and `-0.30 relation_recovery`.
  - Cross-architecture KV/latent linear transfer: **REFUTED** (`PPL = 493,744`, `EM = 0.0`).

---

## 2. Compatibility Matrix

| Capability | Integration Status | Mechanism / Adapter Needed | Notes / Constraints |
|---|---|---|---|
| **CMCP TLV frames over `VectLinkRing`** | **Direct via Adapter** | `vectlink_adapter` + `cmcp_adapter` using `QuantScheme.RAW` | SPSC ring; `WriteSlot` + `ReadSlot` gives single-copy SHM transport with CRC32C. |
| **Multi-reader / persistent state snapshots in `KVArena`** | **Via Adapter** | Store CMCP state snapshots or KV blocks in `KVArena` (`l3_dir` optional) | Enables multiple consumers & process restart recovery when `l3_dir` is configured. |
| **Same-model KV Reuse (`KVArena` + `llama.cpp`)** | **Conditional (Same Model Only)** | C/C++ bridge (`llama_state_seq_get_data` / `set_data`) + `KVIdentity` check | Requires exact match of weights, arch, tokenizer, RoPE, dtype, and `llama.cpp` ABI. |
| **Cross-architecture KV Injection** | **FORBIDDEN / NOT INTEGRABLE** | Rejected by `KVIdentity` mismatch | Empirically refuted in CMCP 2.1; `KVIdentity` enforces strict incompatibility rejection. |

---

## 3. Risks & Portability Constraints

1. **Windows vs. POSIX SHM Lifetime:** On Windows (`CreateFileMapping`), kernel SHM objects are destroyed when the **last open handle** closes (`vtlk_shm_unlink` is a no-op). Therefore, in multi-process tests on Windows, either the consumer/broker must hold a handle open before the producer exits, or `KVArena(..., l3_dir=...)` must be used for cold restart recovery across disjoint process lifetimes.
2. **SPSC Ring Concurrency:** `VectLinkRing` is strictly single-producer, single-consumer. Multi-agent fan-out requires one ring per `(producer, consumer)` pair or `KVArena` sealed blocks for multi-reader broadcast.
3. **ABI & Decoupling:** `VectLink-CMCP` must consume `Vect-Link` via its public `vectlink` Python package / `vectlink.dll` C ABI and `CMCP` via `CMCPStateStore` / `encode_frame` / `decode_frame` without modifying either upstream repo or creating circular dependencies.
4. **No End-to-End Zero-Copy Claim for Semantic LLM Context:** Even when `VectLinkRing` / `KVArena` avoids intermediate OS pipe copies in SHM, `CMCPStateStore.render_for_llm()` still serializes text for LLM tokenization/prefill. We will never claim end-to-end zero-copy for text-rendered semantic context.
