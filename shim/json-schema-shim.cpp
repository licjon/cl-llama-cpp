#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <cstdlib>

using json = nlohmann::ordered_json;

extern "C" {

struct llama_json_schema_result {
    char *output;
    int   status;   // 0 = ok, 1 = json parse error, 2 = conversion error, 3 = other
};

struct llama_json_schema_result llama_json_schema_to_grammar(
        const char *json_schema_str, int force_gbnf) {
    struct llama_json_schema_result result = {nullptr, 0};
    try {
        json schema = json::parse(json_schema_str);
        std::string grammar = json_schema_to_grammar(schema, force_gbnf != 0);
        result.output = strdup(grammar.c_str());
        result.status = 0;
    } catch (const json::parse_error &e) {
        result.output = strdup(e.what());
        result.status = 1;
    } catch (const std::invalid_argument &e) {
        result.output = strdup(e.what());
        result.status = 2;
    } catch (const std::exception &e) {
        result.output = strdup(e.what());
        result.status = 3;
    }
    return result;
}

void llama_json_schema_shim_free(char *ptr) {
    free(ptr);
}

} // extern "C"
