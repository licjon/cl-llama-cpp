#include "chat.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <cstdlib>

using json = nlohmann::ordered_json;

struct llama_extras_chat_ctx {
    common_chat_templates *tmpls;
    common_chat_params cached_params;
    bool has_cached_params;
};

extern "C" {

struct llama_extras_chat_result {
    char *output;
    int status; // 0 = ok, 1 = error
};

struct llama_extras_chat_init_result {
    void *handle;
    char *error;
};

struct llama_extras_chat_init_result llama_extras_chat_init(
        const struct llama_model *model,
        const char *template_override,
        const char *bos_override,
        const char *eos_override) {
    struct llama_extras_chat_init_result result = {nullptr, nullptr};
    try {
        std::string tmpl(template_override ? template_override : "");
        std::string bos(bos_override ? bos_override : "");
        std::string eos(eos_override ? eos_override : "");

        auto ptr = common_chat_templates_init(model, tmpl, bos, eos);
        if (!ptr) {
            result.error = strdup("common_chat_templates_init returned NULL");
            return result;
        }

        auto *ctx = new llama_extras_chat_ctx();
        ctx->tmpls = ptr.release();
        ctx->has_cached_params = false;
        result.handle = ctx;
    } catch (const std::exception &e) {
        result.error = strdup(e.what());
    } catch (...) {
        result.error = strdup("unknown error during chat templates init");
    }
    return result;
}

void llama_extras_chat_free(void *handle) {
    if (!handle) return;
    auto *ctx = static_cast<llama_extras_chat_ctx *>(handle);
    if (ctx->tmpls) {
        common_chat_templates_free(ctx->tmpls);
    }
    delete ctx;
}

char *llama_extras_chat_templates_source(const void *handle, const char *variant) {
    if (!handle) return nullptr;
    try {
        auto *ctx = static_cast<const llama_extras_chat_ctx *>(handle);
        std::string v(variant ? variant : "");
        std::string src = common_chat_templates_source(ctx->tmpls, v);
        return strdup(src.c_str());
    } catch (...) {
        return nullptr;
    }
}

struct llama_extras_chat_result llama_extras_chat_apply(
        void *handle,
        const char *messages_json,
        const char *tools_json,
        int tool_choice,
        int add_generation_prompt,
        int use_jinja,
        int parallel_tool_calls,
        int reasoning_format,
        int enable_thinking,
        const char *json_schema_str,
        const char *grammar_str,
        int continue_final_message) {
    struct llama_extras_chat_result result = {nullptr, 1};
    if (!handle) {
        result.output = strdup("handle is NULL");
        return result;
    }
    try {
        auto *ctx = static_cast<llama_extras_chat_ctx *>(handle);

        common_chat_templates_inputs inputs;

        if (messages_json && messages_json[0]) {
            json msgs = json::parse(messages_json);
            inputs.messages = common_chat_msgs_parse_oaicompat(msgs);
        }

        if (tools_json && tools_json[0]) {
            json tools = json::parse(tools_json);
            inputs.tools = common_chat_tools_parse_oaicompat(tools);
        }

        inputs.tool_choice = static_cast<common_chat_tool_choice>(tool_choice);
        inputs.add_generation_prompt = (add_generation_prompt != 0);
        inputs.use_jinja = (use_jinja != 0);
        inputs.parallel_tool_calls = (parallel_tool_calls != 0);
        inputs.reasoning_format = static_cast<common_reasoning_format>(reasoning_format);
        inputs.enable_thinking = (enable_thinking != 0);
        inputs.continue_final_message = static_cast<common_chat_continuation>(continue_final_message);

        if (json_schema_str && json_schema_str[0]) {
            inputs.json_schema = json_schema_str;
        }
        if (grammar_str && grammar_str[0]) {
            inputs.grammar = grammar_str;
        }

        ctx->cached_params = common_chat_templates_apply(ctx->tmpls, inputs);
        ctx->has_cached_params = true;

        const auto &cp = ctx->cached_params;

        json out;
        out["prompt"] = cp.prompt;
        out["grammar"] = cp.grammar;
        out["format"] = static_cast<int>(cp.format);
        out["grammar_lazy"] = cp.grammar_lazy;
        out["generation_prompt"] = cp.generation_prompt;
        out["supports_thinking"] = cp.supports_thinking;
        out["thinking_start_tag"] = cp.thinking_start_tag;
        out["thinking_end_tag"] = cp.thinking_end_tag;

        json stops = json::array();
        for (const auto &s : cp.additional_stops) stops.push_back(s);
        out["additional_stops"] = stops;

        json preserved = json::array();
        for (const auto &t : cp.preserved_tokens) preserved.push_back(t);
        out["preserved_tokens"] = preserved;

        json triggers = json::array();
        for (const auto &t : cp.grammar_triggers) {
            json tr;
            tr["type"] = static_cast<int>(t.type);
            tr["value"] = t.value;
            tr["token"] = t.token;
            triggers.push_back(tr);
        }
        out["grammar_triggers"] = triggers;

        result.output = strdup(out.dump().c_str());
        result.status = 0;
    } catch (const std::exception &e) {
        result.output = strdup(e.what());
        result.status = 1;
    } catch (...) {
        result.output = strdup("unknown error during chat apply");
        result.status = 1;
    }
    return result;
}

struct llama_extras_chat_result llama_extras_chat_parse_cached(
        void *handle,
        const char *input,
        int is_partial,
        int reasoning_format) {
    struct llama_extras_chat_result result = {nullptr, 1};
    if (!handle) {
        result.output = strdup("handle is NULL");
        return result;
    }
    try {
        auto *ctx = static_cast<llama_extras_chat_ctx *>(handle);
        if (!ctx->has_cached_params) {
            result.output = strdup("no cached params - call apply first");
            return result;
        }

        common_chat_parser_params pp(ctx->cached_params);
        pp.reasoning_format = static_cast<common_reasoning_format>(reasoning_format);

        std::string input_str(input ? input : "");
        common_chat_msg msg = common_chat_parse(input_str, is_partial != 0, pp);

        json out = msg.to_json_oaicompat();
        if (!msg.reasoning_content.empty() && !out.contains("reasoning_content")) {
            out["reasoning_content"] = msg.reasoning_content;
        }

        result.output = strdup(out.dump().c_str());
        result.status = 0;
    } catch (const std::exception &e) {
        result.output = strdup(e.what());
        result.status = 1;
    } catch (...) {
        result.output = strdup("unknown error during chat parse");
        result.status = 1;
    }
    return result;
}

struct llama_extras_chat_result llama_extras_chat_parse_simple(
        const char *input,
        int is_partial,
        int format,
        int reasoning_format,
        int parse_tool_calls,
        const char *generation_prompt) {
    struct llama_extras_chat_result result = {nullptr, 1};
    try {
        common_chat_parser_params pp;
        pp.format = static_cast<common_chat_format>(format);
        pp.reasoning_format = static_cast<common_reasoning_format>(reasoning_format);
        pp.parse_tool_calls = (parse_tool_calls != 0);
        if (generation_prompt && generation_prompt[0]) {
            pp.generation_prompt = generation_prompt;
        }

        std::string input_str(input ? input : "");
        common_chat_msg msg = common_chat_parse(input_str, is_partial != 0, pp);

        json out = msg.to_json_oaicompat();
        if (!msg.reasoning_content.empty() && !out.contains("reasoning_content")) {
            out["reasoning_content"] = msg.reasoning_content;
        }

        result.output = strdup(out.dump().c_str());
        result.status = 0;
    } catch (const std::exception &e) {
        result.output = strdup(e.what());
        result.status = 1;
    } catch (...) {
        result.output = strdup("unknown error during chat parse");
        result.status = 1;
    }
    return result;
}

struct llama_extras_chat_result llama_extras_chat_format_single(
        void *handle,
        const char *past_messages_json,
        const char *new_message_json,
        int add_assistant,
        int use_jinja) {
    struct llama_extras_chat_result result = {nullptr, 1};
    if (!handle) {
        result.output = strdup("handle is NULL");
        return result;
    }
    try {
        auto *ctx = static_cast<llama_extras_chat_ctx *>(handle);

        std::vector<common_chat_msg> past_msgs;
        if (past_messages_json && past_messages_json[0]) {
            json past = json::parse(past_messages_json);
            past_msgs = common_chat_msgs_parse_oaicompat(past);
        }

        common_chat_msg new_msg;
        if (new_message_json && new_message_json[0]) {
            json nmj = json::parse(new_message_json);
            json arr = json::array();
            arr.push_back(nmj);
            auto msgs = common_chat_msgs_parse_oaicompat(arr);
            if (!msgs.empty()) {
                new_msg = msgs[0];
            }
        }

        std::string out = common_chat_format_single(
            ctx->tmpls, past_msgs, new_msg,
            add_assistant != 0, use_jinja != 0);

        result.output = strdup(out.c_str());
        result.status = 0;
    } catch (const std::exception &e) {
        result.output = strdup(e.what());
        result.status = 1;
    } catch (...) {
        result.output = strdup("unknown error during chat format single");
        result.status = 1;
    }
    return result;
}

int llama_extras_chat_verify_template(const char *tmpl, int use_jinja) {
    if (!tmpl || !tmpl[0]) return 0;
    try {
        std::string tmpl_str(tmpl);
        return common_chat_verify_template(tmpl_str, use_jinja != 0) ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

void llama_extras_chat_shim_free(char *ptr) {
    free(ptr);
}

} // extern "C"
