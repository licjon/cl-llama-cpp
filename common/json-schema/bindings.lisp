(in-package #:cl-llama-cpp/common/json-schema)

(cffi:defcstruct json-schema-result
  (output :pointer)
  (status :int))

(cffi:defcfun ("llama_json_schema_to_grammar" %json-schema-to-grammar)
    (:struct json-schema-result)
  (json-schema-str :string)
  (force-gbnf :int))

(cffi:defcfun ("llama_json_schema_shim_free" %shim-free) :void
  (ptr :pointer))
