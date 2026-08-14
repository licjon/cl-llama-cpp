(in-package #:cl-llama-cpp/common/speculative)

;;; Params builder
(cffi:defcfun ("llama_extras_spec_params_create" %params-create) :pointer)
(cffi:defcfun ("llama_extras_spec_params_free" %params-free) :void
  (p :pointer))
(cffi:defcfun ("llama_extras_spec_params_add_type" %params-add-type) :void
  (p :pointer) (type :int))
(cffi:defcfun ("llama_extras_spec_params_set_draft_n_max" %params-set-draft-n-max) :void
  (p :pointer) (n :int32))
(cffi:defcfun ("llama_extras_spec_params_set_draft_n_min" %params-set-draft-n-min) :void
  (p :pointer) (n :int32))
(cffi:defcfun ("llama_extras_spec_params_set_draft_p_min" %params-set-draft-p-min) :void
  (p :pointer) (val :float))
(cffi:defcfun ("llama_extras_spec_params_set_draft_p_split" %params-set-draft-p-split) :void
  (p :pointer) (val :float))
(cffi:defcfun ("llama_extras_spec_params_set_ngram_n" %params-set-ngram-n) :void
  (p :pointer) (v :uint16))
(cffi:defcfun ("llama_extras_spec_params_set_ngram_m" %params-set-ngram-m) :void
  (p :pointer) (v :uint16))
(cffi:defcfun ("llama_extras_spec_params_set_ngram_min_hits" %params-set-ngram-min-hits) :void
  (p :pointer) (v :uint16))
(cffi:defcfun ("llama_extras_spec_n_max" %spec-n-max) :int32
  (p :pointer))

;;; Lifecycle
(cffi:defcstruct spec-result
  (ctx :pointer)
  (error :pointer))

(cffi:defcfun ("llama_extras_spec_init" %spec-init) (:struct spec-result)
  (p :pointer) (n-seq :unsigned-int))
(cffi:defcfun ("llama_extras_spec_free" %spec-free) :void
  (ctx :pointer))

;;; Operations
(cffi:defcfun ("llama_extras_spec_begin" %spec-begin) :void
  (ctx :pointer) (seq-id :int)
  (prompt-tokens :pointer) (n-prompt :int32))

(cffi:defcfun ("llama_extras_spec_dp_set_n_past" %dp-set-n-past) :void
  (ctx :pointer) (seq-id :int) (n-past :int32))
(cffi:defcfun ("llama_extras_spec_dp_set_id_last" %dp-set-id-last) :void
  (ctx :pointer) (seq-id :int) (token-id :int32))
(cffi:defcfun ("llama_extras_spec_dp_set_drafting" %dp-set-drafting) :void
  (ctx :pointer) (seq-id :int) (enable :int))
(cffi:defcfun ("llama_extras_spec_dp_set_n_max" %dp-set-n-max) :void
  (ctx :pointer) (seq-id :int) (n-max :int32))
(cffi:defcfun ("llama_extras_spec_dp_set_prompt" %dp-set-prompt) :void
  (ctx :pointer) (seq-id :int) (tokens :pointer) (n :int32))
(cffi:defcfun ("llama_extras_spec_dp_prepare_result" %dp-prepare-result) :void
  (ctx :pointer) (seq-id :int))
(cffi:defcfun ("llama_extras_spec_dp_get_result" %dp-get-result) :int32
  (ctx :pointer) (seq-id :int) (out-buf :pointer) (buf-size :int32))

(cffi:defcfun ("llama_extras_spec_draft" %spec-draft) :void
  (ctx :pointer))
(cffi:defcfun ("llama_extras_spec_process" %spec-process) :int
  (ctx :pointer) (batch-ptr :pointer))
(cffi:defcfun ("llama_extras_spec_accept" %spec-accept) :void
  (ctx :pointer) (seq-id :int) (n-accepted :uint16))
(cffi:defcfun ("llama_extras_spec_need_embd" %spec-need-embd) :int
  (ctx :pointer))
(cffi:defcfun ("llama_extras_spec_need_embd_nextn" %spec-need-embd-nextn) :int
  (ctx :pointer))
(cffi:defcfun ("llama_extras_spec_print_stats" %spec-print-stats) :void
  (ctx :pointer))

(cffi:defcfun ("llama_extras_shim_free" %shim-free) :void
  (ptr :pointer))
