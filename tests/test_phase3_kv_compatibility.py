"""Phase 3 KV Compatibility Gate Tests (Section 5.1)."""

from __future__ import annotations

import pytest

from vectlink_cmcp.kv_reuse_bridge import (
    KVIncompatibleError,
    KVModelSpec,
    assert_kv_compatible,
    check_kv_compatibility,
)

QWEN_05B = KVModelSpec(
    model_name="qwen2.5-0.5b-instruct-q4_k_m.gguf",
    weights_sha256="74a4da8c9fdbcd15",
    arch="qwen2",
    tokenizer_hash="qwen2-bpe-151936",
    n_layer=24,
    n_head_kv=2,
    head_dim=64,
    kv_dtype="f16",
    rope_config_hash="rope-1000000",
    runtime_version="llama.cpp-b4000",
)

LLAMA_1B = KVModelSpec(
    model_name="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    weights_sha256="6f85a640a97cf2bf",
    arch="llama",
    tokenizer_hash="llama3-bpe-128256",
    n_layer=16,
    n_head_kv=8,
    head_dim=64,
    kv_dtype="f16",
    rope_config_hash="rope-500000-llama3",
    runtime_version="llama.cpp-b4000",
)


def test_same_model_is_kv_compatible() -> None:
    ok, reason = check_kv_compatibility(QWEN_05B, QWEN_05B)
    assert ok is True
    assert "compatible" in reason.lower()
    assert_kv_compatible(QWEN_05B, QWEN_05B)


def test_cross_arch_qwen_vs_llama_rejected() -> None:
    ok, reason = check_kv_compatibility(QWEN_05B, LLAMA_1B)
    assert ok is False
    assert "weights_sha256" in reason
    assert "arch" in reason
    with pytest.raises(KVIncompatibleError):
        assert_kv_compatible(QWEN_05B, LLAMA_1B)


def test_dimension_match_with_different_weights_or_rope_rejected() -> None:
    same_dims_diff_rope = KVModelSpec(
        model_name="qwen2.5-0.5b-finetune",
        weights_sha256="deadbeef",
        arch="qwen2",
        tokenizer_hash="qwen2-bpe-151936",
        n_layer=24,
        n_head_kv=2,
        head_dim=64,
        kv_dtype="f16",
        rope_config_hash="rope-different",
        runtime_version="llama.cpp-b4000",
    )
    ok, _ = check_kv_compatibility(QWEN_05B, same_dims_diff_rope)
    assert ok is False
