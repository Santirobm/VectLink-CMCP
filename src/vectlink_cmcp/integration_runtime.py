"""Integration Runtime coordinating CMCP (Level 1) and Vect-Link (Level 2)."""

from __future__ import annotations

import socket
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cmcp_adapter import CMCPAdapter, EnvelopeMetadata, StateOp
from .vectlink_adapter import VectLinkRingTransport


class ConventionalSocketTransport:
    """Conventional loopback TCP IPC baseline (length-prefixed frames) for comparison."""

    def __init__(self, conn: socket.socket) -> None:
        self._conn = conn

    def publish(self, data: bytes, sequence: int = 0) -> None:
        _ = sequence
        hdr = struct.pack("<I", len(data))
        self._conn.sendall(hdr + data)

    def consume_blocking(self) -> bytes:
        hdr = self._recv_exact(4)
        (length,) = struct.unpack("<I", hdr)
        return self._recv_exact(length)

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self._conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed during IPC read")
            buf.extend(chunk)
        return bytes(buf)

    def close(self) -> None:
        self._conn.close()


class ProducerSession:
    """Agent A session: applies local StateOps and publishes CMCP envelopes over transport."""

    def __init__(
        self,
        adapter: CMCPAdapter,
        transport: Any,
    ) -> None:
        self.adapter = adapter
        self.transport = transport

    def commit_and_publish(self, ops: Sequence[StateOp]) -> bytes:
        self.adapter.apply_local_ops(ops)
        seq = max((op.seq for op in ops), default=self.adapter.last_seq)
        pkt = self.adapter.encode_ops(ops, seq=seq)
        self.transport.publish(pkt, sequence=seq)
        return pkt


class ConsumerSession:
    """Agent B session: receives envelopes, validates integrity/version, applies idempotently."""

    def __init__(
        self,
        adapter: CMCPAdapter,
        transport: Any,
    ) -> None:
        self.adapter = adapter
        self.transport = transport
        self.seen_seqs: set[int] = set()

    def poll_and_apply_ring(self) -> Optional[Tuple[EnvelopeMetadata, List[str]]]:
        assert isinstance(self.transport, VectLinkRingTransport)
        rec = self.transport.try_consume()
        if rec is None:
            return None
        meta, statuses = self.adapter.decode_and_apply(rec.payload)
        self.seen_seqs.add(meta.seq)
        return meta, statuses

    def consume_socket_and_apply(self) -> Tuple[EnvelopeMetadata, List[str]]:
        assert isinstance(self.transport, ConventionalSocketTransport)
        pkt = self.transport.consume_blocking()
        meta, statuses = self.adapter.decode_and_apply(pkt)
        self.seen_seqs.add(meta.seq)
        return meta, statuses

    def render_context(self, style: str = "r3") -> str:
        return self.adapter.render(style=style)

    def checkout_history(self, target_seq: int) -> Dict[str, Dict[str, Any]]:
        return self.adapter.checkout_seq(target_seq)
