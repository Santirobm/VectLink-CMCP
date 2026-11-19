"""Phase 3 KV Reuse Compatibility Gate & Bridge (Level 2 Optimization).

Strictly enforces all compatibility conditions (Section 5.1) before allowing
any KV-cache reuse across processes:
  - model_id / weights_sha256
  - architecture & tokenizer_hash
  - kv_dtype, n_layer, n_head_kv, head_dim
  - rope_config_hash & runtime_version

Never attempts cross-architecture KV injection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


class KVIncompatibleError(ValueError):
    """Raised when two models/runtimes do not satisfy strict KV compatibility."""


@dataclass(frozen=True)
class KVModelSpec:
    model_name: str
    weights_sha256: str
    arch: str
    tokenizer_hash: str
    n_layer: int
    n_head_kv: int
    head_dim: int
    kv_dtype: str
    rope_config_hash: str
    runtime_version: str


def check_kv_compatibility(producer: KVModelSpec, consumer: KVModelSpec) -> Tuple[bool, str]:
    """Return (compatible, reason). Requires exact match across all 9 fields."""
    checks = [
        ("weights_sha256", producer.weights_sha256, consumer.weights_sha256),
        ("arch", producer.arch, consumer.arch),
        ("tokenizer_hash", producer.tokenizer_hash, consumer.tokenizer_hash),
        ("n_layer", producer.n_layer, consumer.n_layer),
        ("n_head_kv", producer.n_head_kv, consumer.n_head_kv),
        ("head_dim", producer.head_dim, consumer.head_dim),
        ("kv_dtype", producer.kv_dtype, consumer.kv_dtype),
        ("rope_config_hash", producer.rope_config_hash, consumer.rope_config_hash),
        ("runtime_version", producer.runtime_version, consumer.runtime_version),
    ]
    mismatches = [f"{name} ({a!r} != {b!r})" for name, a, b in checks if a != b]
    if mismatches:
        return False, "KV incompatible: " + "; ".join(mismatches)
    return True, "KV compatible (exact model, tokenizer, RoPE, dtype, and runtime match)"


def assert_kv_compatible(producer: KVModelSpec, consumer: KVModelSpec) -> None:
    ok, reason = check_kv_compatibility(producer, consumer)
    if not ok:
        raise KVIncompatibleError(reason)
