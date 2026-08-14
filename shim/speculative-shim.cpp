#include "speculative.h"
#include <cstdlib>
#include <cstring>
#include <vector>

struct llama_extras_spec_params {
    common_params_speculative params;
};

struct llama_extras_spec_context {
    common_speculative * spec;
    uint32_t n_seq;
    std::vector<llama_tokens> prompts;
    std::vector<llama_tokens> results;
};

extern "C" {

// --- Params builder ---

llama_extras_spec_params * llama_extras_spec_params_create(void) {
    try {
        return new llama_extras_spec_params();
    } catch (...) {
        return nullptr;
    }
}

void llama_extras_spec_params_free(llama_extras_spec_params * p) {
    delete p;
}

void llama_extras_spec_params_add_type(llama_extras_spec_params * p, int type) {
    if (!p) return;
    auto & types = p->params.types;
    if (types.size() == 1 && types[0] == COMMON_SPECULATIVE_TYPE_NONE) {
        types.clear();
    }
    types.push_back(static_cast<common_speculative_type>(type));
}

void llama_extras_spec_params_set_draft_n_max(
        llama_extras_spec_params * p, int32_t n) {
    if (p) p->params.draft.n_max = n;
}

void llama_extras_spec_params_set_draft_n_min(
        llama_extras_spec_params * p, int32_t n) {
    if (p) p->params.draft.n_min = n;
}

void llama_extras_spec_params_set_draft_p_min(
        llama_extras_spec_params * p, float val) {
    if (p) p->params.draft.p_min = val;
}

void llama_extras_spec_params_set_draft_p_split(
        llama_extras_spec_params * p, float val) {
    if (p) p->params.draft.p_split = val;
}

void llama_extras_spec_params_set_ngram_n(
        llama_extras_spec_params * p, uint16_t v) {
    if (p) p->params.ngram_simple.size_n = v;
}

void llama_extras_spec_params_set_ngram_m(
        llama_extras_spec_params * p, uint16_t v) {
    if (p) p->params.ngram_simple.size_m = v;
}

void llama_extras_spec_params_set_ngram_min_hits(
        llama_extras_spec_params * p, uint16_t v) {
    if (p) p->params.ngram_simple.min_hits = v;
}

// --- Lifecycle ---

struct llama_extras_spec_result {
    void * ctx;
    char * error;
};

struct llama_extras_spec_result llama_extras_spec_init(
        llama_extras_spec_params * p, uint32_t n_seq) {
    struct llama_extras_spec_result result = {nullptr, nullptr};
    if (!p) {
        result.error = strdup("params is NULL");
        return result;
    }
    try {
        common_speculative * spec = common_speculative_init(p->params, n_seq);
        if (!spec) {
            result.error = strdup("common_speculative_init returned NULL");
            return result;
        }
        auto * ctx = new llama_extras_spec_context();
        ctx->spec = spec;
        ctx->n_seq = n_seq;
        ctx->prompts.resize(n_seq);
        ctx->results.resize(n_seq);
        result.ctx = ctx;
    } catch (const std::exception & e) {
        result.error = strdup(e.what());
    } catch (...) {
        result.error = strdup("unknown error during speculative init");
    }
    return result;
}

void llama_extras_spec_free(void * ctx_ptr) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    if (ctx->spec) {
        common_speculative_free(ctx->spec);
    }
    delete ctx;
}

// --- Operations ---

void llama_extras_spec_begin(void * ctx_ptr, int seq_id,
                             const int32_t * prompt_tokens, int32_t n_prompt) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    llama_tokens prompt(prompt_tokens, prompt_tokens + n_prompt);
    try {
        common_speculative_begin(ctx->spec, seq_id, prompt);
    } catch (...) {}
}

void llama_extras_spec_dp_set_n_past(void * ctx_ptr, int seq_id,
                                      int32_t n_past) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        auto & dp = common_speculative_get_draft_params(ctx->spec, seq_id);
        dp.n_past = n_past;
    } catch (...) {}
}

void llama_extras_spec_dp_set_id_last(void * ctx_ptr, int seq_id,
                                       int32_t token_id) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        auto & dp = common_speculative_get_draft_params(ctx->spec, seq_id);
        dp.id_last = token_id;
    } catch (...) {}
}

void llama_extras_spec_dp_set_drafting(void * ctx_ptr, int seq_id,
                                        int enable) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        auto & dp = common_speculative_get_draft_params(ctx->spec, seq_id);
        dp.drafting = (enable != 0);
    } catch (...) {}
}

void llama_extras_spec_dp_set_n_max(void * ctx_ptr, int seq_id,
                                     int32_t n_max) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        auto & dp = common_speculative_get_draft_params(ctx->spec, seq_id);
        dp.n_max = n_max;
    } catch (...) {}
}

void llama_extras_spec_dp_set_prompt(void * ctx_ptr, int seq_id,
                                      const int32_t * tokens, int32_t n) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    if (seq_id < 0 || (uint32_t)seq_id >= ctx->n_seq) return;
    ctx->prompts[seq_id].assign(tokens, tokens + n);
    try {
        auto & dp = common_speculative_get_draft_params(ctx->spec, seq_id);
        dp.prompt = &ctx->prompts[seq_id];
    } catch (...) {}
}

void llama_extras_spec_dp_prepare_result(void * ctx_ptr, int seq_id) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    if (seq_id < 0 || (uint32_t)seq_id >= ctx->n_seq) return;
    ctx->results[seq_id].clear();
    try {
        auto & dp = common_speculative_get_draft_params(ctx->spec, seq_id);
        dp.result = &ctx->results[seq_id];
    } catch (...) {}
}

void llama_extras_spec_draft(void * ctx_ptr) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        common_speculative_draft(ctx->spec);
    } catch (...) {}
}

int32_t llama_extras_spec_dp_get_result(void * ctx_ptr, int seq_id,
                                         int32_t * out_buf,
                                         int32_t buf_size) {
    if (!ctx_ptr) return 0;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    if (seq_id < 0 || (uint32_t)seq_id >= ctx->n_seq) return 0;
    const auto & result = ctx->results[seq_id];
    int32_t n = static_cast<int32_t>(result.size());
    int32_t copy_n = (n < buf_size) ? n : buf_size;
    if (out_buf && copy_n > 0) {
        memcpy(out_buf, result.data(), copy_n * sizeof(int32_t));
    }
    return n;
}

int llama_extras_spec_process(void * ctx_ptr, const void * batch_ptr) {
    if (!ctx_ptr || !batch_ptr) return 0;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    const auto * batch = static_cast<const llama_batch *>(batch_ptr);
    try {
        return common_speculative_process(ctx->spec, *batch) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

void llama_extras_spec_accept(void * ctx_ptr, int seq_id,
                               uint16_t n_accepted) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        common_speculative_accept(ctx->spec, seq_id, n_accepted);
    } catch (...) {}
}

int llama_extras_spec_need_embd(void * ctx_ptr) {
    if (!ctx_ptr) return 0;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        return common_speculative_need_embd(ctx->spec) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int llama_extras_spec_need_embd_nextn(void * ctx_ptr) {
    if (!ctx_ptr) return 0;
    auto * ctx = static_cast<llama_extras_spec_context *>(ctx_ptr);
    try {
        return common_speculative_need_embd_nextn(ctx->spec) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

void llama_extras_spec_print_stats(const void * ctx_ptr) {
    if (!ctx_ptr) return;
    auto * ctx = static_cast<const llama_extras_spec_context *>(ctx_ptr);
    try {
        common_speculative_print_stats(ctx->spec);
    } catch (...) {}
}

int32_t llama_extras_spec_n_max(llama_extras_spec_params * p) {
    if (!p) return 0;
    try {
        return common_speculative_n_max(&p->params);
    } catch (...) {
        return 0;
    }
}

void llama_extras_shim_free(char * ptr) {
    free(ptr);
}

} // extern "C"
