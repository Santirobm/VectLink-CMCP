"""CMCP Adapter (Level 1 — Semantic Memory).

Wraps Cross-Model-Context-Protocol (CMCP) state management, versioned envelopes,
TLV serialization, integrity checks, and historical reconstruction without any
dependency on Vect-Link.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _ensure_cmcp_path() -> Path:
    env_root = os.environ.get("CMCP_ROOT")
    if env_root:
        root = Path(env_root).resolve()
    else:
        root = Path(__file__).resolve().parents[3] / "Cross-Model-Context-Protocol"
    if not root.exists():
        raise FileNotFoundError(f"CMCP repository not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_ensure_cmcp_path()

from representations.binary import (  # noqa: E402
    CMCP_MSG_CONTEXT_TRANSFER,
    CMCP_TIER_2_CROSS_MODEL,
    decode_frame,
    encode_frame,
)
from representations.state_store import CMCPStateStore, StateOp  # noqa: E402


ENVELOPE_MAGIC = b"VCM1"
SCHEMA_VERSION = 1
_ENVELOPE_FMT = "<4sHQ16sII"
_ENVELOPE_SIZE = struct.calcsize(_ENVELOPE_FMT)


class CMCPAdapterError(Exception):
    """Base error for CMCP adapter validation failures."""


class IncompatibleSchemaError(CMCPAdapterError):
    """Raised when an incoming envelope has an unsupported schema version."""


class PayloadCorruptError(CMCPAdapterError):
    """Raised when envelope length, magic, or checksum validation fails."""


@dataclass(frozen=True)
class EnvelopeMetadata:
    schema_version: int
    seq: int
    state_id: str
    payload_len: int
    checksum: int


def _state_id_bytes(state_id: str) -> bytes:
    raw = state_id.encode("utf-8")
    if len(raw) <= 16:
        return raw.ljust(16, b"\x00")
    return hashlib.sha256(raw).digest()[:16]


def pack_ops_to_tlv(ops: Sequence[StateOp]) -> List[Tuple[int, bytes]]:
    recs: List[Tuple[int, bytes]] = []
    for op in ops:
        blob = json.dumps(
            {
                "id": op.fact_id,
                "op": op.kind,
                "cat": op.category,
                "k": op.key,
                "v": op.value,
                "s": op.seq,
                "ver": op.version,
                "ag": op.agent_id,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        tag = (
            0x03
            if op.category == "relation"
            else (0x05 if op.category == "constraint" else (0x08 if op.category == "decision" else 0x02))
        )
        recs.append((tag, blob))
    return recs


def unpack_ops_from_tlv(recs: Sequence[Tuple[int, bytes]]) -> List[StateOp]:
    out: List[StateOp] = []
    for _, val in recs:
        d = json.loads(val.decode("utf-8"))
        out.append(
            StateOp(
                fact_id=d["id"],
                kind=d["op"],
                category=d["cat"],
                key=d["k"],
                value=d["v"],
                seq=int(d["s"]),
                version=int(d["ver"]),
                agent_id=d.get("ag", "agent_a"),
            )
        )
    return out


class CMCPAdapter:
    """Manages versioned CMCP semantic state and validated wire envelopes."""

    def __init__(self, state_id: str = "default-session", conflict_policy: str = "lww") -> None:
        self.state_id = state_id
        self.store = CMCPStateStore(conflict_policy=conflict_policy)
        self.last_seq = 0
        self.applied_ops = 0
        self.ignored_duplicates = 0
        self.rejected_stale = 0

    def apply_local_ops(self, ops: Sequence[StateOp]) -> List[str]:
        statuses: List[str] = []
        for op in ops:
            st = self.store.apply(op)
            statuses.append(st)
            if st == "APPLIED":
                self.applied_ops += 1
                self.last_seq = max(self.last_seq, op.seq)
            elif st == "DUPLICATE_IGNORED":
                self.ignored_duplicates += 1
            elif st in ("STALE_IGNORED", "STALE_REJECTED", "TOMBSTONE_BLOCKED"):
                self.rejected_stale += 1
        return statuses

    def encode_ops(self, ops: Sequence[StateOp], seq: Optional[int] = None) -> bytes:
        """Serialize StateOps into a CMCP TLV frame wrapped in a versioned envelope."""
        tlv_recs = pack_ops_to_tlv(ops)
        tlv_frame = encode_frame(
            tlv_recs,
            compress=True,
            msg_type=CMCP_MSG_CONTEXT_TRANSFER,
            compat_tier=CMCP_TIER_2_CROSS_MODEL,
        )
        eff_seq = seq if seq is not None else max((op.seq for op in ops), default=self.last_seq)
        sid_bytes = _state_id_bytes(self.state_id)
        crc = zlib.crc32(tlv_frame) & 0xFFFFFFFF
        hdr = struct.pack(
            _ENVELOPE_FMT,
            ENVELOPE_MAGIC,
            SCHEMA_VERSION,
            eff_seq,
            sid_bytes,
            len(tlv_frame),
            crc,
        )
        return hdr + tlv_frame

    @staticmethod
    def inspect_envelope(packet: bytes) -> Tuple[EnvelopeMetadata, bytes]:
        if len(packet) < _ENVELOPE_SIZE:
            raise PayloadCorruptError(f"Truncated envelope: {len(packet)} < {_ENVELOPE_SIZE}")
        magic, ver, seq, sid_raw, plen, crc = struct.unpack(_ENVELOPE_FMT, packet[:_ENVELOPE_SIZE])
        if magic != ENVELOPE_MAGIC:
            raise PayloadCorruptError(f"Invalid envelope magic: {magic!r}")
        if ver != SCHEMA_VERSION:
            raise IncompatibleSchemaError(f"Unsupported schema_version={ver}, expected {SCHEMA_VERSION}")
        body = packet[_ENVELOPE_SIZE:]
        if len(body) != plen:
            raise PayloadCorruptError(f"Payload length mismatch: got {len(body)}, expected {plen}")
        actual_crc = zlib.crc32(body) & 0xFFFFFFFF
        if actual_crc != crc:
            raise PayloadCorruptError(f"Envelope checksum mismatch: {actual_crc:#x} != {crc:#x}")
        sid = sid_raw.rstrip(b"\x00").decode("utf-8", errors="replace")
        return EnvelopeMetadata(ver, seq, sid, plen, crc), body

    def decode_and_apply(self, packet: bytes) -> Tuple[EnvelopeMetadata, List[str]]:
        """Validate envelope + TLV CRC32C and apply ops idempotently to CMCPStateStore."""
        meta, tlv_frame = self.inspect_envelope(packet)
        try:
            _, recs = decode_frame(tlv_frame)
        except Exception as exc:
            raise PayloadCorruptError(f"TLV frame decode failed: {exc}") from exc
        ops = unpack_ops_from_tlv(recs)
        statuses = self.apply_local_ops(ops)
        return meta, statuses

    def render(self, style: str = "r3") -> str:
        return self.store.render_for_llm(style=style)

    def active_view(self) -> Dict[str, Dict[str, Any]]:
        return self.store.active_view()

    def checkout_seq(self, target_seq: int) -> Dict[str, Dict[str, Any]]:
        return self.store.checkout_seq(target_seq=target_seq)

    def compact(self, prune_tombstones_before_seq: Optional[int] = None) -> int:
        return self.store.compact(prune_tombstones_before_seq=prune_tombstones_before_seq)
