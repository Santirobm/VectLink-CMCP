"""Vect-Link Adapter (Level 2 — Shared Memory Transport & KV Arena).

Interacts strictly with Vect-Link's public Python/C APIs (VectLinkRing, WriteSlot,
ReadSlot, KVArena, KVIdentity, Branch, Snapshot) without touching private internals
or importing CMCP.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def _ensure_vectlink_path() -> Path:
    env_root = os.environ.get("VECTLINK_ROOT")
    if env_root:
        root = Path(env_root).resolve()
    else:
        root = Path(__file__).resolve().parents[3] / "Vect-Link"
    if not root.exists():
        raise FileNotFoundError(f"Vect-Link repository not found at {root}")
    py_dir = root / "python"
    if str(py_dir) not in sys.path:
        sys.path.insert(0, str(py_dir))
    if not os.environ.get("VECTLINK_LIB_PATH"):
        for cand in (
            root / "build" / "vectlink.dll",
            root / "build" / "libvectlink.so",
            root / "build" / "libvectlink.dylib",
        ):
            if cand.exists():
                os.environ["VECTLINK_LIB_PATH"] = str(cand)
                break
    return root


_ensure_vectlink_path()

from vectlink import (  # noqa: E402
    Branch,
    CrcMismatchError,
    KVArena,
    KVDtype,
    KVIdentity,
    QuantScheme,
    RingBusyError,
    RingEmptyError,
    RingFullError,
    ShmError,
    Snapshot,
    VectLinkError,
    VectLinkRing,
)


@dataclass(frozen=True)
class ReceivedPacket:
    payload: bytes
    sequence: int
    crc32c: int


class VectLinkRingTransport:
    """SPSC shared-memory ring transport using Vect-Link public WriteSlot/ReadSlot."""

    def __init__(self, ring: VectLinkRing, is_owner: bool) -> None:
        self._ring = ring
        self._is_owner = is_owner

    @classmethod
    def create(cls, shm_name: str, capacity: int = 16, max_payload_bytes: int = 65536) -> "VectLinkRingTransport":
        ring = VectLinkRing.create(shm_name, slot_capacity=max_payload_bytes, slot_count=capacity)
        return cls(ring, is_owner=True)

    @classmethod
    def open(cls, shm_name: str, capacity: int = 16, max_payload_bytes: int = 65536) -> "VectLinkRingTransport":
        _ = capacity, max_payload_bytes
        ring = VectLinkRing.open(shm_name)
        return cls(ring, is_owner=False)

    def publish(self, data: bytes, sequence: int = 0) -> None:
        with self._ring.acquire_write() as slot:
            slot.write_bytes(data)
            slot.commit(
                payload_bytes=len(data),
                quant=QuantScheme.RAW,
                sequence=sequence,
            )

    def try_consume(self) -> Optional[ReceivedPacket]:
        try:
            with self._ring.acquire_read() as slot:
                hdr = slot.header
                data = bytes(slot.buffer)
                slot.release()
                return ReceivedPacket(
                    payload=data,
                    sequence=hdr.sequence,
                    crc32c=hdr.payload_crc32c,
                )
        except RingEmptyError:
            return None

    def close(self, unlink: bool = False) -> None:
        self._ring.close()
        if unlink and self._is_owner:
            self._ring.unlink()


class VectLinkArenaTransport:
    """Multi-reader CoW KVArena wrapper for sealed snapshots & optional L3 cold recovery."""

    def __init__(self, arena: KVArena, is_owner: bool) -> None:
        self._arena = arena
        self._is_owner = is_owner

    @classmethod
    def create(
        cls,
        shm_name: str,
        num_blocks: int = 32,
        block_size_bytes: int = 65536,
    ) -> "VectLinkArenaTransport":
        arena = KVArena.create(
            shm_name,
            num_blocks=num_blocks,
            block_size_bytes=block_size_bytes,
        )
        return cls(arena, is_owner=True)

    @classmethod
    def open(
        cls,
        shm_name: str,
        num_blocks: int = 32,
        block_size_bytes: int = 65536,
    ) -> "VectLinkArenaTransport":
        _ = num_blocks, block_size_bytes
        arena = KVArena.open(shm_name)
        return cls(arena, is_owner=False)

    @staticmethod
    def make_identity(seq: int, payload: bytes, arch_hash: int = 0x434D4350) -> KVIdentity:
        return KVIdentity(
            token_prefix_hash=KVArena.hash_bytes(payload),
            model_arch_hash=arch_hash,
            rope_config_hash=1,
            lora_adapter_hash=0,
            pos_offset=seq,
            token_count=len(payload),
            kv_dtype=KVDtype.RAW_LLAMA,
            layer_count=1,
            head_dim=64,
            kv_head_count=1,
        )

    def publish_block(self, seq: int, payload: bytes, arch_hash: int = 0x434D4350) -> Tuple[KVIdentity, int]:
        ident = self.make_identity(seq, payload, arch_hash=arch_hash)
        with self._arena.alloc_block() as blk:
            blk.write_bytes(payload)
            bid = blk.seal(ident)
        return ident, bid

    def read_block(self, ident: KVIdentity) -> Optional[bytes]:
        bid = self._arena.lookup_block(ident)
        if bid is None:
            return None
        with self._arena.open_block(bid) as blk:
            return bytes(blk.buffer[: ident.token_count])

    def stats(self):
        return self._arena.stats()

    def close(self, unlink: bool = False) -> None:
        self._arena.close()
        if unlink and self._is_owner:
            self._arena.unlink()
