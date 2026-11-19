#define _CRT_SECURE_NO_WARNINGS
#include "llama.h"
#include "vectlink_kv.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <time.h>
#endif

#define KV_BLOCK_BYTES 262144u
#define KV_ARENA_BLOCKS 256u

static void quiet_log_cb(enum ggml_log_level level, const char *text, void *user_data) {
    (void)level; (void)text; (void)user_data;
}

static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER cnt;
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static double get_rss_mib(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) return (double)ru.ru_maxrss / 1024.0;
    return 0.0;
#endif
}

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return buf;
}

static void print_json_escaped(const char *s) {
    putchar('"');
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"') printf("\\\"");
        else if (c == '\\') printf("\\\\");
        else if (c == '\n') printf("\\n");
        else if (c == '\r') printf("\\r");
        else if (c == '\t') printf("\\t");
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

static int tokenize_text(const struct llama_vocab *vocab, const char *text, bool add_bos, llama_token **out_toks, int32_t *out_n) {
    int32_t len = (int32_t)strlen(text);
    int32_t cap = len + 16;
    llama_token *toks = (llama_token *)malloc((size_t)cap * sizeof(llama_token));
    if (!toks) return -1;
    int32_t n = llama_tokenize(vocab, text, len, toks, cap, add_bos, true);
    if (n < 0) {
        cap = -n + 4;
        free(toks);
        toks = (llama_token *)malloc((size_t)cap * sizeof(llama_token));
        if (!toks) return -1;
        n = llama_tokenize(vocab, text, len, toks, cap, add_bos, true);
    }
    if (n < 0) { free(toks); return -1; }
    *out_toks = toks;
    *out_n = n;
    return 0;
}

static vtlk_kv_identity_t build_model_identity(const struct llama_model *model, uint32_t token_count) {
    vtlk_kv_identity_t id;
    memset(&id, 0, sizeof(id));
    id.token_prefix_hash = 0x56434D50ULL;
    id.model_arch_hash = (uint64_t)llama_model_n_params(model) ^ ((uint64_t)llama_model_n_embd(model) << 16);
    id.rope_config_hash = (uint64_t)(uint32_t)llama_model_rope_type(model) + 1ULL;
    id.lora_adapter_hash = 0;
    id.pos_offset = 0;
    id.token_count = token_count;
    id.kv_dtype = VTLK_KV_DTYPE_RAW_LLAMA;
    id.layer_count = (uint32_t)llama_model_n_layer(model);
    id.head_dim = (uint32_t)(llama_model_n_embd(model) / (llama_model_n_head(model) > 0 ? llama_model_n_head(model) : 1));
    id.kv_head_count = (uint32_t)llama_model_n_head_kv(model);
    return id;
}

static int decode_greedy(struct llama_context *ctx, const struct llama_vocab *vocab, int32_t n_past, int32_t max_tokens, char *out_buf, size_t out_cap, double *out_ttft_from_start_ms, double t_start, double *out_decode_ms) {
    struct llama_sampler *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    size_t pos = 0;
    out_buf[0] = '\0';
    double t_dec0 = now_ms();
    bool got_first = false;

    for (int32_t step = 0; step < max_tokens; ++step) {
        llama_token tok = llama_sampler_sample(smpl, ctx, -1);
        if (!got_first) {
            *out_ttft_from_start_ms = now_ms() - t_start;
            got_first = true;
        }
        if (llama_vocab_is_eog(vocab, tok)) break;
        char piece[256];
        int n = llama_token_to_piece(vocab, tok, piece, sizeof(piece) - 1, 0, true);
        if (n > 0 && pos + (size_t)n < out_cap) {
            memcpy(out_buf + pos, piece, (size_t)n);
            pos += (size_t)n;
            out_buf[pos] = '\0';
        }
        struct llama_batch b = llama_batch_init(1, 0, 1);
        b.n_tokens = 1;
        b.token[0] = tok;
        b.pos[0] = n_past + step;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 0;
        b.logits[0] = 1;
        if (llama_decode(ctx, b) != 0) {
            llama_batch_free(b);
            break;
        }
        llama_batch_free(b);
    }
    if (!got_first) *out_ttft_from_start_ms = now_ms() - t_start;
    *out_decode_ms = now_ms() - t_dec0;
    llama_sampler_free(smpl);
    return 0;
}

typedef struct kv_blob_hdr {
    uint64_t kv_bytes;
    uint32_t n_prefix_tokens;
    uint32_t n_blocks;
} kv_blob_hdr_t;

int main(int argc, char **argv) {
    const char *model_path = NULL, *shm_name = "/vcmcp_kv_arena", *role = "cold";
    const char *prefix_file = NULL, *query_file = NULL;
    int32_t n_ctx = 1024, threads = 4, max_tokens = 32;
    uint64_t snapshot_id = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--shm") == 0 && i + 1 < argc) shm_name = argv[++i];
        else if (strcmp(argv[i], "--role") == 0 && i + 1 < argc) role = argv[++i];
        else if (strcmp(argv[i], "--prefix-file") == 0 && i + 1 < argc) prefix_file = argv[++i];
        else if (strcmp(argv[i], "--query-file") == 0 && i + 1 < argc) query_file = argv[++i];
        else if (strcmp(argv[i], "--n-ctx") == 0 && i + 1 < argc) n_ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--snapshot-id") == 0 && i + 1 < argc) snapshot_id = (uint64_t)strtoull(argv[++i], NULL, 10);
    }
    if (!model_path) { fprintf(stderr, "Missing --model\n"); return 2; }

    llama_log_set(quiet_log_cb, NULL);
    llama_backend_init();

    double t_all0 = now_ms();
    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 3; }

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = (uint32_t)n_ctx;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 4; }
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    double init_ms = now_ms() - t_all0;

    if (strcmp(role, "producer") == 0) {
        char *ptxt = prefix_file ? read_text_file(prefix_file) : strdup("");
        llama_token *ptoks = NULL;
        int32_t n_ptoks = 0;
        tokenize_text(vocab, ptxt, true, &ptoks, &n_ptoks);
        double t_pre0 = now_ms();
        struct llama_batch b = llama_batch_init(n_ptoks, 0, 1);
        b.n_tokens = n_ptoks;
        for (int32_t i = 0; i < n_ptoks; ++i) {
            b.token[i] = ptoks[i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == n_ptoks - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "Producer decode failed\n"); return 5; }
        llama_batch_free(b);
        double prefill_ms = now_ms() - t_pre0;

        double t_ser0 = now_ms();
        size_t kv_sz = llama_state_seq_get_size(ctx, 0);
        uint8_t *kv_buf = (uint8_t *)malloc(kv_sz);
        size_t kv_written = llama_state_seq_get_data(ctx, kv_buf, kv_sz, 0);

        vtlk_kv_arena_t arena;
        if (vtlk_kv_arena_create(&arena, shm_name, KV_ARENA_BLOCKS, KV_BLOCK_BYTES) != VTLK_OK) {
            fprintf(stderr, "Arena create failed\n"); return 6;
        }
        size_t total_blob = sizeof(kv_blob_hdr_t) + kv_written;
        uint32_t n_blocks = (uint32_t)((total_blob + KV_BLOCK_BYTES - 1) / KV_BLOCK_BYTES);
        uint8_t *blob = (uint8_t *)calloc(n_blocks, KV_BLOCK_BYTES);
        kv_blob_hdr_t *bh = (kv_blob_hdr_t *)blob;
        bh->kv_bytes = (uint64_t)kv_written;
        bh->n_prefix_tokens = (uint32_t)n_ptoks;
        bh->n_blocks = n_blocks;
        memcpy(blob + sizeof(kv_blob_hdr_t), kv_buf, kv_written);

        vtlk_kv_identity_t id = build_model_identity(model, (uint32_t)n_ptoks);
        vtlk_kv_branch_t br;
        memset(&br, 0, sizeof(br));
        br.model_arch_hash = id.model_arch_hash;
        br.rope_config_hash = id.rope_config_hash;
        br.lora_adapter_hash = id.lora_adapter_hash;
        br.kv_dtype = id.kv_dtype;
        br.layer_count = id.layer_count;
        br.head_dim = id.head_dim;
        br.kv_head_count = id.kv_head_count;
        for (uint32_t bi = 0; bi < n_blocks; ++bi) {
            uint32_t bid = 0;
            if (vtlk_kv_block_alloc(&arena, &bid) != VTLK_OK) { fprintf(stderr, "Block alloc failed\n"); return 7; }
            memcpy(vtlk_kv_block_data(&arena, bid), blob + (size_t)bi * KV_BLOCK_BYTES, KV_BLOCK_BYTES);
            vtlk_kv_identity_t bid_id = id;
            bid_id.pos_offset = bi;
            if (vtlk_kv_block_seal(&arena, bid, &bid_id) != VTLK_OK) { fprintf(stderr, "Block seal failed\n"); return 8; }
            vtlk_kv_branch_append_block(&arena, &br, bid);
        }
        uint64_t sid = 0;
        if (vtlk_kv_snapshot_create(&arena, &br, &sid) != VTLK_OK) { fprintf(stderr, "Snapshot failed\n"); return 9; }
        double ser_ms = now_ms() - t_ser0;

        printf("{\"ready\":true,\"snapshot_id\":%llu,\"kv_bytes\":%zu,\"n_prefix_tokens\":%d,\"n_blocks\":%u,\"prefill_ms\":%.3f,\"ser_ms\":%.3f,\"rss_mib\":%.2f}\n",
               (unsigned long long)sid, kv_written, n_ptoks, n_blocks, prefill_ms, ser_ms, get_rss_mib());
        fflush(stdout);

        int ch;
        while ((ch = getchar()) != EOF) {}
        vtlk_kv_branch_release(&arena, &br);
        vtlk_kv_arena_close(&arena);
        free(blob); free(kv_buf); free(ptoks); free(ptxt);
    } else if (strcmp(role, "consumer") == 0) {
        char *qtxt = query_file ? read_text_file(query_file) : strdup("");
        llama_token *qtoks = NULL;
        int32_t n_qtoks = 0;
        tokenize_text(vocab, qtxt, false, &qtoks, &n_qtoks);

        double t_res0 = now_ms();
        vtlk_kv_arena_t arena;
        if (vtlk_kv_arena_open(&arena, shm_name) != VTLK_OK) { fprintf(stderr, "Arena open failed\n"); return 10; }
        vtlk_kv_identity_t expected = build_model_identity(model, 0);
        vtlk_kv_branch_t br;
        const void *ptrs[VTLK_KV_MAX_BRANCH_BLOCKS];
        memset(&br, 0, sizeof(br));
        if (vtlk_kv_restore(&arena, snapshot_id, &expected, &br, ptrs, VTLK_KV_MAX_BRANCH_BLOCKS) != VTLK_OK) {
            fprintf(stderr, "KV restore/identity check failed\n"); return 11;
        }
        const kv_blob_hdr_t *bh = (const kv_blob_hdr_t *)ptrs[0];
        size_t kv_bytes = (size_t)bh->kv_bytes;
        uint32_t n_ptoks = bh->n_prefix_tokens;
        uint8_t *kv_buf = (uint8_t *)malloc(kv_bytes);
        size_t off = 0;
        for (uint32_t bi = 0; bi < br.block_count && off < kv_bytes; ++bi) {
            const uint8_t *src = (const uint8_t *)ptrs[bi];
            size_t skip = (bi == 0) ? sizeof(kv_blob_hdr_t) : 0;
            size_t avail = KV_BLOCK_BYTES - skip;
            size_t take = (kv_bytes - off < avail) ? (kv_bytes - off) : avail;
            memcpy(kv_buf + off, src + skip, take);
            off += take;
        }
        size_t set_res = llama_state_seq_set_data(ctx, kv_buf, kv_bytes, 0);
        free(kv_buf);
        if (set_res == 0) { fprintf(stderr, "llama_state_seq_set_data failed\n"); return 12; }
        double restore_ms = now_ms() - t_res0;

        double t_pre0 = now_ms();
        struct llama_batch b = llama_batch_init(n_qtoks, 0, 1);
        b.n_tokens = n_qtoks;
        for (int32_t i = 0; i < n_qtoks; ++i) {
            b.token[i] = qtoks[i];
            b.pos[i] = (int32_t)n_ptoks + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == n_qtoks - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "Consumer query decode failed\n"); return 13; }
        llama_batch_free(b);
        double prefill_ms = now_ms() - t_pre0;

        char out_buf[4096];
        double ttft_ms = 0.0, decode_ms = 0.0;
        decode_greedy(ctx, vocab, (int32_t)n_ptoks + n_qtoks, max_tokens, out_buf, sizeof(out_buf), &ttft_ms, t_all0, &decode_ms);
        double e2e_ms = now_ms() - t_all0;

        printf("{\"role\":\"consumer\",\"init_ms\":%.3f,\"restore_ms\":%.3f,\"prefill_ms\":%.3f,\"ttft_ms\":%.3f,\"decode_ms\":%.3f,\"e2e_ms\":%.3f,\"rss_mib\":%.2f,\"kv_bytes\":%zu,\"n_prefix_tokens\":%u,\"n_query_tokens\":%d,\"output_text\":",
               init_ms, restore_ms, prefill_ms, ttft_ms, decode_ms, e2e_ms, get_rss_mib(), kv_bytes, n_ptoks, n_qtoks);
        print_json_escaped(out_buf);
        printf("}\n");

        vtlk_kv_branch_release(&arena, &br);
        vtlk_kv_arena_close(&arena);
        free(qtoks); free(qtxt);
    } else {
        /* cold prefill: prefix + query */
        char *ptxt = prefix_file ? read_text_file(prefix_file) : strdup("");
        char *qtxt = query_file ? read_text_file(query_file) : strdup("");
        size_t full_len = strlen(ptxt) + strlen(qtxt) + 1;
        char *full = (char *)malloc(full_len);
        strcpy(full, ptxt);
        strcat(full, qtxt);

        llama_token *toks = NULL;
        int32_t n_toks = 0;
        tokenize_text(vocab, full, true, &toks, &n_toks);

        double t_pre0 = now_ms();
        struct llama_batch b = llama_batch_init(n_toks, 0, 1);
        b.n_tokens = n_toks;
        for (int32_t i = 0; i < n_toks; ++i) {
            b.token[i] = toks[i];
            b.pos[i] = i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = (i == n_toks - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "Cold decode failed\n"); return 14; }
        llama_batch_free(b);
        double prefill_ms = now_ms() - t_pre0;

        char out_buf[4096];
        double ttft_ms = 0.0, decode_ms = 0.0;
        decode_greedy(ctx, vocab, n_toks, max_tokens, out_buf, sizeof(out_buf), &ttft_ms, t_all0, &decode_ms);
        double e2e_ms = now_ms() - t_all0;

        printf("{\"role\":\"cold\",\"init_ms\":%.3f,\"restore_ms\":0.0,\"prefill_ms\":%.3f,\"ttft_ms\":%.3f,\"decode_ms\":%.3f,\"e2e_ms\":%.3f,\"rss_mib\":%.2f,\"kv_bytes\":0,\"n_prefix_tokens\":%d,\"n_query_tokens\":0,\"output_text\":",
               init_ms, prefill_ms, ttft_ms, decode_ms, e2e_ms, get_rss_mib(), n_toks);
        print_json_escaped(out_buf);
        printf("}\n");

        free(toks); free(full); free(ptxt); free(qtxt);
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
