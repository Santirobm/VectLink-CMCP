#include "vectlink_kv.h"
#include "vectlink/transport.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "CHECK FAILED at line %d: %s\n", __LINE__, #cond); return 1; } } while (0)

int main(void) {
    vtlk_shm_t shm;
    vtlk_ring_t ring;
    memset(&shm, 0, sizeof(shm));
    memset(&ring, 0, sizeof(ring));
    const char *name = "/vcmcp_c_test_ring";
    size_t req = vtlk_ring_required_bytes(4, 1024, NULL);
    CHECK(req > 0);
    CHECK(vtlk_shm_create(&shm, name, req) == VTLK_OK);
    CHECK(vtlk_ring_init(&ring, shm.addr, req, 4, 1024) == VTLK_OK);

    const char *msg = "VCM1-C-API-OK";
    vtlk_write_slot_t ws;
    CHECK(vtlk_ring_acquire_write(&ring, &ws) == VTLK_OK);
    memcpy(ws.payload, msg, strlen(msg));
    vtlk_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "VTLK", 4);
    hdr.header_bytes = 64;
    hdr.version = 1;
    hdr.d_model = 1;
    hdr.n_tokens = 1;
    hdr.raw_bytes = (uint32_t)strlen(msg);
    hdr.quant_scheme = 3; /* RAW */
    hdr.sequence = 42;
    CHECK(vtlk_ring_commit_write(&ring, &ws, &hdr, strlen(msg)) == VTLK_OK);

    vtlk_read_slot_t rs;
    CHECK(vtlk_ring_acquire_read(&ring, &rs) == VTLK_OK);
    CHECK(rs.bytes == strlen(msg));
    CHECK(rs.header.sequence == 42);
    CHECK(memcmp(rs.payload, msg, strlen(msg)) == 0);
    CHECK(vtlk_ring_release_read(&ring, &rs) == VTLK_OK);

    CHECK(vtlk_ring_close(&ring) == VTLK_OK);
    vtlk_shm_close(&shm);
    vtlk_shm_unlink(name);
    printf("test_vl_cmcp_c_api PASSED\n");
    return 0;
}
