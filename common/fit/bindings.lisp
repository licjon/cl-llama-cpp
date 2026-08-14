(in-package #:cl-llama-cpp/common/fit)

;;; Lifecycle
(cffi:defcfun ("llama_extras_fit_create" %fit-create) :pointer)
(cffi:defcfun ("llama_extras_fit_free" %fit-free) :void
  (ptr :pointer))
(cffi:defcfun ("llama_extras_fit_run" %fit-run) :int
  (ptr :pointer) (path-model :string)
  (margin :uint64) (n-ctx-min :uint32) (log-level :int))

;;; Result queries
(cffi:defcfun ("llama_extras_fit_status" %fit-status) :int
  (ptr :pointer))
(cffi:defcfun ("llama_extras_fit_n_gpu_layers" %fit-n-gpu-layers) :int32
  (ptr :pointer))
(cffi:defcfun ("llama_extras_fit_n_ctx" %fit-n-ctx) :uint32
  (ptr :pointer))
(cffi:defcfun ("llama_extras_fit_n_devices" %fit-n-devices) :int32)
(cffi:defcfun ("llama_extras_fit_tensor_split_at" %fit-tensor-split-at) :float
  (ptr :pointer) (index :int32))

;;; Print functions
(cffi:defcfun ("llama_extras_fit_print" %fit-print) :int
  (path-model :string))
(cffi:defcfun ("llama_extras_fit_memory_breakdown_print"
               %fit-memory-breakdown-print) :void
  (ctx :pointer))
