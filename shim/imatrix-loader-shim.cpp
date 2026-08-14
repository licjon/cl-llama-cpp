#include "imatrix-loader.h"
#include <vector>
#include <string>

struct imatrix_wrapper {
    common_imatrix imatrix;
    std::vector<std::string> cached_keys;

    void cache_keys() {
        cached_keys.clear();
        cached_keys.reserve(imatrix.entries.size());
        for (const auto & e : imatrix.entries) {
            cached_keys.push_back(e.first);
        }
    }
};

extern "C" {

void * llama_extras_imatrix_create() {
    try {
        return new imatrix_wrapper();
    } catch (...) {
        return nullptr;
    }
}

void llama_extras_imatrix_free(void * ptr) {
    if (!ptr) return;
    delete static_cast<imatrix_wrapper *>(ptr);
}

int llama_extras_imatrix_load(void * ptr, const char * fname) {
    if (!ptr || !fname) return 0;
    auto * w = static_cast<imatrix_wrapper *>(ptr);
    try {
        bool ok = common_imatrix_load(std::string(fname), w->imatrix);
        if (ok) {
            w->cache_keys();
        }
        return ok ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int32_t llama_extras_imatrix_chunk_count(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<const imatrix_wrapper *>(ptr)->imatrix.chunk_count;
}

int32_t llama_extras_imatrix_chunk_size(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<const imatrix_wrapper *>(ptr)->imatrix.chunk_size;
}

int llama_extras_imatrix_is_legacy(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<const imatrix_wrapper *>(ptr)->imatrix.is_legacy ? 1 : 0;
}

int llama_extras_imatrix_has_metadata(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<const imatrix_wrapper *>(ptr)->imatrix.has_metadata ? 1 : 0;
}

int32_t llama_extras_imatrix_n_datasets(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<int32_t>(
        static_cast<const imatrix_wrapper *>(ptr)->imatrix.datasets.size());
}

const char * llama_extras_imatrix_dataset(const void * ptr, int32_t index) {
    if (!ptr) return nullptr;
    auto * w = static_cast<const imatrix_wrapper *>(ptr);
    if (index < 0 || index >= static_cast<int32_t>(w->imatrix.datasets.size()))
        return nullptr;
    return w->imatrix.datasets[index].c_str();
}

int32_t llama_extras_imatrix_n_entries(const void * ptr) {
    if (!ptr) return 0;
    return static_cast<int32_t>(
        static_cast<const imatrix_wrapper *>(ptr)->cached_keys.size());
}

const char * llama_extras_imatrix_entry_name(const void * ptr, int32_t index) {
    if (!ptr) return nullptr;
    auto * w = static_cast<const imatrix_wrapper *>(ptr);
    if (index < 0 || index >= static_cast<int32_t>(w->cached_keys.size()))
        return nullptr;
    return w->cached_keys[index].c_str();
}

int32_t llama_extras_imatrix_entry_n_sums(const void * ptr, int32_t index) {
    if (!ptr) return 0;
    auto * w = static_cast<const imatrix_wrapper *>(ptr);
    if (index < 0 || index >= static_cast<int32_t>(w->cached_keys.size()))
        return 0;
    auto it = w->imatrix.entries.find(w->cached_keys[index]);
    if (it == w->imatrix.entries.end()) return 0;
    return static_cast<int32_t>(it->second.sums.size());
}

const float * llama_extras_imatrix_entry_sums(const void * ptr, int32_t index) {
    if (!ptr) return nullptr;
    auto * w = static_cast<const imatrix_wrapper *>(ptr);
    if (index < 0 || index >= static_cast<int32_t>(w->cached_keys.size()))
        return nullptr;
    auto it = w->imatrix.entries.find(w->cached_keys[index]);
    if (it == w->imatrix.entries.end()) return nullptr;
    return it->second.sums.data();
}

int32_t llama_extras_imatrix_entry_n_counts(const void * ptr, int32_t index) {
    if (!ptr) return 0;
    auto * w = static_cast<const imatrix_wrapper *>(ptr);
    if (index < 0 || index >= static_cast<int32_t>(w->cached_keys.size()))
        return 0;
    auto it = w->imatrix.entries.find(w->cached_keys[index]);
    if (it == w->imatrix.entries.end()) return 0;
    return static_cast<int32_t>(it->second.counts.size());
}

const int64_t * llama_extras_imatrix_entry_counts(const void * ptr, int32_t index) {
    if (!ptr) return nullptr;
    auto * w = static_cast<const imatrix_wrapper *>(ptr);
    if (index < 0 || index >= static_cast<int32_t>(w->cached_keys.size()))
        return nullptr;
    auto it = w->imatrix.entries.find(w->cached_keys[index]);
    if (it == w->imatrix.entries.end()) return nullptr;
    return it->second.counts.data();
}

} // extern "C"
