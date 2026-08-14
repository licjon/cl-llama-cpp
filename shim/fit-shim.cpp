#include "fit.h"
#include <vector>
#include <cstring>

struct fit_wrapper {
    llama_model_params mparams;
    llama_context_params cparams;
    std::vector<float> tensor_split_buf;
    std::vector<llama_model_tensor_buft_override> overrides_buf;
    std::vector<size_t> margins_buf;
    common_params_fit_status status;

    fit_wrapper() {
        mparams = llama_model_default_params();
        cparams = llama_context_default_params();
        cparams.n_ctx = 0;

        size_t nd = llama_max_devices();
        tensor_split_buf.resize(nd, 0.0f);

        size_t ntbo = llama_max_tensor_buft_overrides();
        overrides_buf.resize(ntbo);
        memset(overrides_buf.data(), 0, ntbo * sizeof(llama_model_tensor_buft_override));

        margins_buf.resize(nd > 0 ? nd : 1, 0);

        status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
};

extern "C" {

void * llama_extras_fit_create() {
    try {
        return new fit_wrapper();
    } catch (...) {
        return nullptr;
    }
}

void llama_extras_fit_free(void * ptr) {
    if (!ptr) return;
    delete static_cast<fit_wrapper *>(ptr);
}

int llama_extras_fit_run(void * ptr, const char * path_model,
                         uint64_t margin, uint32_t n_ctx_min, int log_level) {
    if (!ptr || !path_model) return COMMON_PARAMS_FIT_STATUS_ERROR;
    auto * w = static_cast<fit_wrapper *>(ptr);

    for (auto & m : w->margins_buf) {
        m = static_cast<size_t>(margin);
    }

    try {
        w->status = common_fit_params(
            path_model,
            &w->mparams,
            &w->cparams,
            w->tensor_split_buf.data(),
            w->overrides_buf.data(),
            w->margins_buf.data(),
            n_ctx_min,
            static_cast<ggml_log_level>(log_level));
    } catch (...) {
        w->status = COMMON_PARAMS_FIT_STATUS_ERROR;
    }
    return static_cast<int>(w->status);
}

int llama_extras_fit_status(const void * ptr) {
    if (!ptr) return COMMON_PARAMS_FIT_STATUS_ERROR;
    return static_cast<int>(static_cast<const fit_wrapper *>(ptr)->status);
}

int32_t llama_extras_fit_n_gpu_layers(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<const fit_wrapper *>(ptr)->mparams.n_gpu_layers;
}

uint32_t llama_extras_fit_n_ctx(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<const fit_wrapper *>(ptr)->cparams.n_ctx;
}

int32_t llama_extras_fit_n_devices() {
    return static_cast<int32_t>(llama_max_devices());
}

float llama_extras_fit_tensor_split_at(const void * ptr, int32_t index) {
    if (!ptr) return 0.0f;
    auto * w = static_cast<const fit_wrapper *>(ptr);
    if (index < 0 || index >= static_cast<int32_t>(w->tensor_split_buf.size()))
        return 0.0f;
    return w->tensor_split_buf[index];
}

int llama_extras_fit_print(const char * path_model) {
    if (!path_model) return 1;
    try {
        llama_model_params mparams = llama_model_default_params();
        llama_context_params cparams = llama_context_default_params();
        common_fit_print(path_model, &mparams, &cparams);
        return 0;
    } catch (...) {
        return 1;
    }
}

void llama_extras_fit_memory_breakdown_print(const void * ctx) {
    if (!ctx) return;
    try {
        common_memory_breakdown_print(static_cast<const llama_context *>(ctx));
    } catch (...) {
    }
}

} // extern "C"
