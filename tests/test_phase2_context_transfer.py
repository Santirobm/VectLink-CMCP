"""Phase 2 Mandatory Test Suite (Section 4.1): All 15 Context Transfer & SHM tests."""

from __future__ import annotations

import multiprocessing
import struct
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from vectlink_cmcp import (
    CMCPAdapter,
    ConsumerSession,
    IncompatibleSchemaError,
    PayloadCorruptError,
    ProducerSession,
    StateOp,
    VectLinkArenaTransport,
    VectLinkRingTransport,
)


def _shm_name(prefix: str) -> str:
    return f"/vcmcp_{prefix}_{uuid.uuid4().hex[:10]}"


def _mp_consumer_read_once(shm_name: str, cap: int, max_bytes: int, q: multiprocessing.Queue, ready_evt) -> None:
    ring = VectLinkRingTransport.open(shm_name, capacity=cap, max_payload_bytes=max_bytes)
    ready_evt.set()
    try:
        cons = ConsumerSession(CMCPAdapter("mp-cons"), ring)
        while True:
            res = cons.poll_and_apply_ring()
            if res is not None:
                meta, statuses = res
                q.put((meta.seq, statuses, cons.adapter.active_view()))
                break
    finally:
        ring.close()


def test_01_initial_state() -> None:
    shm = _shm_name("t01")
    prod_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    cons_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
    try:
        prod = ProducerSession(CMCPAdapter("s1"), prod_ring)
        cons = ConsumerSession(CMCPAdapter("s1"), cons_ring)
        prod.commit_and_publish([
            StateOp("e1", "UPSERT", "entity", "Node-1", "primary", seq=1, version=1),
            StateOp("c1", "UPSERT", "constraint", "budget", "budget<=50ms", seq=2, version=1),
        ])
        res = cons.poll_and_apply_ring()
        assert res is not None
        meta, statuses = res
        assert meta.seq == 2
        assert statuses == ["APPLIED", "APPLIED"]
        assert "Node-1" in cons.render_context("r3")
    finally:
        cons_ring.close()
        prod_ring.close(unlink=True)


def test_02_incremental_update() -> None:
    shm = _shm_name("t02")
    prod_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    cons_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
    try:
        prod = ProducerSession(CMCPAdapter("s2"), prod_ring)
        cons = ConsumerSession(CMCPAdapter("s2"), cons_ring)
        prod.commit_and_publish([StateOp("f1", "UPSERT", "fact", "route", "R-100", seq=1, version=1)])
        cons.poll_and_apply_ring()
        prod.commit_and_publish([StateOp("f2", "UPSERT", "fact", "tag", "T-200", seq=2, version=1)])
        cons.poll_and_apply_ring()
        view = cons.adapter.active_view()
        assert view["f1"]["value"] == "R-100"
        assert view["f2"]["value"] == "T-200"
    finally:
        cons_ring.close()
        prod_ring.close(unlink=True)


def test_03_duplicate_update_idempotent() -> None:
    shm = _shm_name("t03")
    prod_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    cons_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
    try:
        prod = ProducerSession(CMCPAdapter("s3"), prod_ring)
        cons = ConsumerSession(CMCPAdapter("s3"), cons_ring)
        op = StateOp("f1", "UPSERT", "fact", "route", "R-100", seq=1, version=1)
        pkt = prod.commit_and_publish([op])
        cons.poll_and_apply_ring()
        prod_ring.publish(pkt, sequence=1)
        res2 = cons.poll_and_apply_ring()
        assert res2 is not None
        _, st2 = res2
        assert st2 == ["DUPLICATE_IGNORED"]
        assert cons.adapter.ignored_duplicates == 1
    finally:
        cons_ring.close()
        prod_ring.close(unlink=True)


def test_04_out_of_order_rejected() -> None:
    shm = _shm_name("t04")
    prod_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    cons_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
    try:
        prod = ProducerSession(CMCPAdapter("s4"), prod_ring)
        cons = ConsumerSession(CMCPAdapter("s4"), cons_ring)
        prod.commit_and_publish([StateOp("f1", "UPSERT", "fact", "route", "R-NEW", seq=10, version=2)])
        cons.poll_and_apply_ring()
        stale_pkt = prod.adapter.encode_ops([StateOp("f1", "UPSERT", "fact", "route", "R-OLD", seq=5, version=1)], seq=5)
        prod_ring.publish(stale_pkt, sequence=5)
        _, st = cons.poll_and_apply_ring()  # type: ignore[misc]
        assert st == ["STALE_IGNORED"]
        assert cons.adapter.active_view()["f1"]["value"] == "R-NEW"
    finally:
        cons_ring.close()
        prod_ring.close(unlink=True)


def test_05_and_06_invalidation_and_tombstone_prevents_resurrection() -> None:
    shm = _shm_name("t05")
    prod_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    cons_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
    try:
        prod = ProducerSession(CMCPAdapter("s5"), prod_ring)
        cons = ConsumerSession(CMCPAdapter("s5"), cons_ring)
        prod.commit_and_publish([StateOp("f1", "UPSERT", "fact", "token", "TK-1", seq=1, version=1)])
        cons.poll_and_apply_ring()
        prod.commit_and_publish([StateOp("f1", "INVALIDATE", "fact", "token", "", seq=5, version=2)])
        cons.poll_and_apply_ring()
        assert "f1" not in cons.adapter.active_view()

        # Attempt resurrection at seq=3 (< tombstone seq=5)
        old_pkt = prod.adapter.encode_ops([StateOp("f1", "UPSERT", "fact", "token", "TK-1", seq=3, version=1)], seq=3)
        prod_ring.publish(old_pkt, sequence=3)
        _, st = cons.poll_and_apply_ring()  # type: ignore[misc]
        assert st == ["STALE_IGNORED"]
        assert "f1" not in cons.adapter.active_view()
    finally:
        cons_ring.close()
        prod_ring.close(unlink=True)


def test_07_and_08_fact_correction_and_historical_checkout() -> None:
    shm = _shm_name("t07")
    prod_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    cons_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
    try:
        prod = ProducerSession(CMCPAdapter("s7"), prod_ring)
        cons = ConsumerSession(CMCPAdapter("s7"), cons_ring)
        prod.commit_and_publish([StateOp("c1", "UPSERT", "constraint", "cap", "cap<=400", seq=1, version=1)])
        cons.poll_and_apply_ring()
        prod.commit_and_publish([StateOp("c1", "UPSERT", "constraint", "cap", "cap<=1200", seq=5, version=2)])
        cons.poll_and_apply_ring()
        assert cons.adapter.active_view()["c1"]["value"] == "cap<=1200"
        hist = cons.checkout_history(target_seq=2)
        assert hist["c1"]["value"] == "cap<=400"
    finally:
        cons_ring.close()
        prod_ring.close(unlink=True)


def test_09_consumer_restart_with_shm_arena() -> None:
    shm = _shm_name("t09")
    a_prod = VectLinkArenaTransport.create(shm, num_blocks=8, block_size_bytes=8192)
    try:
        prod = CMCPAdapter("s9")
        pkt = prod.encode_ops([StateOp("d1", "UPSERT", "decision", "db", "SQLite", seq=1, version=1)], seq=1)
        ident, _ = a_prod.publish_block(seq=1, payload=pkt)

        # Consumer 1 attaches, reads, and closes (simulating restart)
        c1 = VectLinkArenaTransport.open(shm)
        raw1 = c1.read_block(ident)
        c1.close()
        assert raw1 == pkt

        # Restarted consumer 2 re-opens SHM arena and reconstructs full state
        c2 = VectLinkArenaTransport.open(shm)
        try:
            recovered = c2.read_block(ident)
            assert recovered == pkt
            cons = CMCPAdapter("s9")
            meta, st = cons.decode_and_apply(recovered)
            assert meta.seq == 1
            assert st == ["APPLIED"]
            assert cons.active_view()["d1"]["value"] == "SQLite"
        finally:
            c2.close()
    finally:
        a_prod.close(unlink=True)


def test_10_producer_restart_resumes_sequence() -> None:
    shm = _shm_name("t10")
    cons_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    try:
        cons = ConsumerSession(CMCPAdapter("s10"), cons_ring)

        # Producer 1 publishes seq=1 then closes
        p1_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
        p1 = ProducerSession(CMCPAdapter("s10"), p1_ring)
        p1.commit_and_publish([StateOp("f1", "UPSERT", "fact", "step", "1", seq=1, version=1)])
        p1_ring.close()
        cons.poll_and_apply_ring()

        # Producer 2 restarts and resumes at seq=2
        p2_ring = VectLinkRingTransport.open(shm, capacity=8, max_payload_bytes=8192)
        p2 = ProducerSession(CMCPAdapter("s10"), p2_ring)
        p2.commit_and_publish([StateOp("f2", "UPSERT", "fact", "step", "2", seq=2, version=1)])
        p2_ring.close()
        cons.poll_and_apply_ring()

        assert cons.adapter.active_view()["f1"]["value"] == "1"
        assert cons.adapter.active_view()["f2"]["value"] == "2"
    finally:
        cons_ring.close(unlink=True)


def test_11_corrupt_payload_rejected() -> None:
    adp = CMCPAdapter("s11")
    pkt = bytearray(adp.encode_ops([StateOp("f1", "UPSERT", "fact", "k", "v", seq=1, version=1)], seq=1))
    pkt[-1] ^= 0xFF
    with pytest.raises(PayloadCorruptError):
        adp.decode_and_apply(bytes(pkt))


def test_12_truncated_payload_rejected() -> None:
    adp = CMCPAdapter("s12")
    pkt = adp.encode_ops([StateOp("f1", "UPSERT", "fact", "k", "v", seq=1, version=1)], seq=1)
    with pytest.raises(PayloadCorruptError):
        adp.decode_and_apply(pkt[:-4])
    with pytest.raises(PayloadCorruptError):
        adp.decode_and_apply(pkt[:8])


def test_13_incompatible_schema_version_rejected() -> None:
    adp = CMCPAdapter("s13")
    pkt = bytearray(adp.encode_ops([StateOp("f1", "UPSERT", "fact", "k", "v", seq=1, version=1)], seq=1))
    # Overwrite schema_version (bytes 4..6) to 99
    struct.pack_into("<H", pkt, 4, 99)
    with pytest.raises(IncompatibleSchemaError):
        adp.decode_and_apply(bytes(pkt))


def test_14_shm_reconnection_across_processes() -> None:
    shm = _shm_name("t14")
    prod_ring = VectLinkRingTransport.create(shm, capacity=8, max_payload_bytes=8192)
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    ready = ctx.Event()
    p = ctx.Process(target=_mp_consumer_read_once, args=(shm, 8, 8192, q, ready))
    p.start()
    try:
        assert ready.wait(timeout=10.0)
        prod = ProducerSession(CMCPAdapter("mp-cons"), prod_ring)
        prod.commit_and_publish([StateOp("x1", "UPSERT", "fact", "mode", "active", seq=7, version=1)])
        seq, statuses, view = q.get(timeout=10.0)
        assert seq == 7
        assert statuses == ["APPLIED"]
        assert view["x1"]["value"] == "active"
        p.join(timeout=10.0)
        assert p.exitcode == 0
    finally:
        if p.is_alive():
            p.terminate()
        prod_ring.close(unlink=True)


def test_15_concurrent_readers_on_kv_arena() -> None:
    shm = _shm_name("t15")
    arena = VectLinkArenaTransport.create(shm, num_blocks=16, block_size_bytes=8192)
    try:
        adp = CMCPAdapter("s15")
        pkt = adp.encode_ops([StateOp("r1", "UPSERT", "relation", "path", "A -> B -> C", seq=1, version=1)], seq=1)
        ident, _ = arena.publish_block(seq=1, payload=pkt)

        def reader_task(_: int) -> str:
            r_arena = VectLinkArenaTransport.open(shm, num_blocks=16, block_size_bytes=8192)
            try:
                raw = r_arena.read_block(ident)
                assert raw is not None
                c = CMCPAdapter("s15")
                c.decode_and_apply(raw)
                return c.active_view()["r1"]["value"]
            finally:
                r_arena.close()

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(reader_task, range(12)))
        assert all(r == "A -> B -> C" for r in results)
    finally:
        arena.close(unlink=True)
