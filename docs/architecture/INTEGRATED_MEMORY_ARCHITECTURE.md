# Integrated Two-Level Memory Architecture: VectLink-CMCP

## 1. Overview & Separation of Concerns

`VectLink-CMCP` investigates a decoupled two-level shared-memory architecture for multi-agent LLM systems:

- **Level 1 — Semantic Memory (`cmcp_adapter`):** Manages facts, entities, relations, decisions, constraints, sequence numbers, tombstones/invalidation, historical checkout (`checkout_seq`), and binary TLV framing (`VCM1` envelope + CMCP TLV + CRC32C).
- **Level 2 — Inference & SHM Transport Memory (`vectlink_adapter`):** Interacts strictly with `Vect-Link` public APIs (`VectLinkRing` SPSC shared-memory ring with `QuantScheme.RAW`, and `KVArena` Copy-on-Write paged block arena with `KVIdentity`, `Branch`, and `Snapshot`).

```
+-----------------------+                    +-----------------------+
|  Process A (Producer) |                    |  Process B (Consumer) |
|  e.g. Qwen2.5-0.5B    |                    |  e.g. Llama-3.2-1B    |
+-----------+-----------+                    +-----------+-----------+
            |                                            ^
            v                                            |
   [ CMCPAdapter (L1) ]                         [ CMCPAdapter (L1) ]
   StateOp -> TLV + VCM1                        Validate VCM1 + Apply
            |                                            ^
            v                                            |
  [ VectLinkAdapter (L2) ] === OS SHM Ring ===> [ VectLinkAdapter (L2) ]
   VectLinkRing / KVArena    (CRC32C + CoW)      ReadSlot / KVArena
            |                                            ^
            +--- Optional Same-Model KV Reuse Only ------+
                 (vtlk_kv_restore + KVIdentity gate)
```

---

## 2. Module Responsibilities

1. **`src/vectlink_cmcp/cmcp_adapter.py`**
   - Zero dependency on `Vect-Link`.
   - Wraps `CMCPStateStore`, `StateOp`, `encode_frame`, `decode_frame`.
   - Adds a 34-byte `VCM1` envelope header (`magic`, `schema_version`, `seq`, `state_id`, `payload_len`, `crc32c`) to enforce schema compatibility and detect truncated/corrupted payloads before TLV decoding.

2. **`src/vectlink_cmcp/vectlink_adapter.py`**
   - Zero dependency on `CMCP`.
   - Wraps `VectLinkRing.create/open`, `WriteSlot.write_bytes/commit`, `ReadSlot.buffer/release` for SPSC SHM transport, and `KVArena` for multi-reader sealed blocks.

3. **`src/vectlink_cmcp/integration_runtime.py`**
   - Coordinates `ProducerSession` and `ConsumerSession` over either `VectLinkRingTransport` or `ConventionalSocketTransport` (loopback TCP baseline).
   - Tracks `seen_seqs` and delegates idempotency/tombstone enforcement to `CMCPStateStore`.

4. **`src/vectlink_cmcp/kv_reuse_bridge.py` + `csrc/vl_cmcp_kv_bridge.c`**
   - Enforces `KVModelSpec` / `vtlk_kv_identity_t` compatibility before any `llama_state_seq_set_data()` restore.
   - Strictly rejects cross-architecture reuse (e.g. `Qwen2.5-0.5B -> Llama-3.2-1B`).

---

## 3. Dependencies & Licenses

- `Vect-Link` (`79ae1a9`, Apache-2.0) — consumed via `vectlink.dll` + `python/vectlink`.
- `Cross-Model-Context-Protocol` (`0a7b649`, Apache-2.0) — consumed via `representations/`.
- `llama.cpp` (MIT) — consumed via `libllama.dll` for `vl_cmcp_kv_bridge.exe` and `cmcp_llama_harness.exe`.
