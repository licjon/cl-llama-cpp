#include "ngram-map.h"
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

// --- ngram simple ---

int32_t llama_extras_ngram_simple_draft(
        uint16_t size_ngram, uint16_t size_mgram,
        const int32_t * tokens, int32_t n_tokens,
        int32_t sampled,
        int32_t * out_buf, int32_t buf_size) {
    try {
        common_ngram_simple_config config;
        config.size_ngram = size_ngram;
        config.size_mgram = size_mgram;
        llama_tokens tok_vec(tokens, tokens + n_tokens);
        llama_tokens draft = common_ngram_simple_draft(config, tok_vec, sampled);
        int32_t n = static_cast<int32_t>(draft.size());
        int32_t copy_n = (n < buf_size) ? n : buf_size;
        if (out_buf && copy_n > 0) {
            memcpy(out_buf, draft.data(), copy_n * sizeof(int32_t));
        }
        return n;
    } catch (...) {
        return 0;
    }
}

// --- ngram map lifecycle ---

void * llama_extras_ngram_map_create(
        uint16_t size_key, uint16_t size_value,
        int key_only, uint16_t min_hits) {
    try {
        return new common_ngram_map(size_key, size_value, key_only != 0, min_hits);
    } catch (...) {
        return nullptr;
    }
}

void llama_extras_ngram_map_free(void * map_ptr) {
    if (!map_ptr) return;
    delete static_cast<common_ngram_map *>(map_ptr);
}

// --- ngram map operations ---

void llama_extras_ngram_map_begin(void * map_ptr,
        const int32_t * tokens, int32_t n_tokens) {
    if (!map_ptr) return;
    auto * map = static_cast<common_ngram_map *>(map_ptr);
    try {
        llama_tokens tok_vec(tokens, tokens + n_tokens);
        common_ngram_map_begin(*map, tok_vec);
    } catch (...) {}
}

int32_t llama_extras_ngram_map_draft(void * map_ptr,
        const int32_t * inp, int32_t n_inp,
        int32_t sampled,
        int32_t * out_buf, int32_t buf_size) {
    if (!map_ptr) return 0;
    auto * map = static_cast<common_ngram_map *>(map_ptr);
    try {
        llama_tokens inp_vec(inp, inp + n_inp);
        llama_tokens draft;
        common_ngram_map_draft(*map, inp_vec, sampled, draft);
        int32_t n = static_cast<int32_t>(draft.size());
        int32_t copy_n = (n < buf_size) ? n : buf_size;
        if (out_buf && copy_n > 0) {
            memcpy(out_buf, draft.data(), copy_n * sizeof(int32_t));
        }
        return n;
    } catch (...) {
        return 0;
    }
}

void llama_extras_ngram_map_accept(void * map_ptr, uint16_t n_accepted) {
    if (!map_ptr) return;
    auto * map = static_cast<common_ngram_map *>(map_ptr);
    try {
        common_ngram_map_accept(*map, n_accepted);
    } catch (...) {}
}

} // extern "C"
