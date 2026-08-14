(in-package #:cl-llama-cpp/common/json-partial)

(cffi:defcstruct json-partial-result
  (json-output :pointer)
  (json-dump-marker :pointer)
  (status :int))

(cffi:defcfun ("llama_extras_json_partial_parse" %json-partial-parse)
    (:struct json-partial-result)
  (input :string)
  (healing-marker :string))

(cffi:defcfun ("llama_extras_json_partial_free" %json-partial-free) :void
  (ptr :pointer))
