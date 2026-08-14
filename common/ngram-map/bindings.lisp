(in-package #:cl-llama-cpp/common/ngram-map)

;;; Simple n-gram draft
(cffi:defcfun ("llama_extras_ngram_simple_draft" %ngram-simple-draft) :int32
  (size-ngram :uint16) (size-mgram :uint16)
  (tokens :pointer) (n-tokens :int32)
  (sampled :int32)
  (out-buf :pointer) (buf-size :int32))

;;; Map lifecycle
(cffi:defcfun ("llama_extras_ngram_map_create" %ngram-map-create) :pointer
  (size-key :uint16) (size-value :uint16)
  (key-only :int) (min-hits :uint16))

(cffi:defcfun ("llama_extras_ngram_map_free" %ngram-map-free) :void
  (map-ptr :pointer))

;;; Map operations
(cffi:defcfun ("llama_extras_ngram_map_begin" %ngram-map-begin) :void
  (map-ptr :pointer) (tokens :pointer) (n-tokens :int32))

(cffi:defcfun ("llama_extras_ngram_map_draft" %ngram-map-draft) :int32
  (map-ptr :pointer)
  (inp :pointer) (n-inp :int32)
  (sampled :int32)
  (out-buf :pointer) (buf-size :int32))

(cffi:defcfun ("llama_extras_ngram_map_accept" %ngram-map-accept) :void
  (map-ptr :pointer) (n-accepted :uint16))
