#include "json-partial.h"
#include <cstring>
#include <cstdlib>

extern "C" {

struct llama_extras_json_partial_result {
    char *json_output;
    char *json_dump_marker;
    int   status; // 0 = ok, 1 = healed, 2 = parse failed, 3 = exception
};

struct llama_extras_json_partial_result llama_extras_json_partial_parse(
        const char *input, const char *healing_marker) {
    struct llama_extras_json_partial_result result = {nullptr, nullptr, 2};
    try {
        common_json out;
        std::string input_str(input ? input : "");
        std::string marker_str(healing_marker ? healing_marker : "");

        bool ok = common_json_parse(input_str, marker_str, out);
        if (!ok) {
            result.status = 2;
            return result;
        }

        result.json_output = strdup(out.json.dump().c_str());
        if (!out.healing_marker.json_dump_marker.empty()) {
            result.json_dump_marker = strdup(out.healing_marker.json_dump_marker.c_str());
            result.status = 1;
        } else {
            result.status = 0;
        }
    } catch (const std::exception &e) {
        free(result.json_output);
        free(result.json_dump_marker);
        result.json_output = strdup(e.what());
        result.json_dump_marker = nullptr;
        result.status = 3;
    }
    return result;
}

void llama_extras_json_partial_free(char *ptr) {
    free(ptr);
}

} // extern "C"
