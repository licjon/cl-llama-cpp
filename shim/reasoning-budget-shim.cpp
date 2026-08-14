#include "reasoning-budget.h"
#include <vector>

extern "C" {

struct llama_sampler * llama_extras_reasoning_budget_init(
        const struct llama_vocab * vocab,
        const int32_t * start_tokens, int32_t n_start,
        const int32_t * end_tokens,   int32_t n_end,
        const int32_t * forced_tokens, int32_t n_forced,
        int32_t budget, int initial_state) {
    try {
        std::vector<llama_token> start(start_tokens, start_tokens + n_start);
        std::vector<llama_token> end(end_tokens, end_tokens + n_end);
        std::vector<llama_token> forced(forced_tokens, forced_tokens + n_forced);
        return common_reasoning_budget_init(
            vocab, start, end, forced, budget,
            static_cast<common_reasoning_budget_state>(initial_state));
    } catch (...) {
        return nullptr;
    }
}

int llama_extras_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    return static_cast<int>(common_reasoning_budget_get_state(smpl));
}

int llama_extras_reasoning_budget_force(struct llama_sampler * smpl) {
    return common_reasoning_budget_force(smpl) ? 1 : 0;
}

} // extern "C"
