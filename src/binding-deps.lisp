(in-package #:cl-llama-cpp)

;;; Every %llama symbol the high-level API depends on.  After regenerating
;;; bindings, call (cl-llama-cpp:check-binding-deps) to verify none were
;;; removed upstream.  The generator also reads this list to flag removals.
(defparameter *binding-deps*
  '(;; Lifecycle
    %llama:backend-init
    %llama:model-default-params %llama:model-load-from-file %llama:model-free
    %llama:context-default-params %llama:new-context-with-model %llama:free
    %llama:get-model %llama:get-memory %llama:memory-clear
    ;; Tokenization
    %llama:model-get-vocab %llama:tokenize %llama:detokenize %llama:token
    %llama:token-bos %llama:token-is-eog
    ;; Generation / batch
    %llama:batch-get-one %llama:batch-init %llama:batch-free
    %llama:decode %llama:encode
    %llama:sampler-chain-default-params %llama:sampler-chain-init
    %llama:sampler-chain-add %llama:sampler-sample %llama:sampler-accept
    %llama:sampler-free
    %llama:sampler-init-greedy %llama:sampler-init-temp
    %llama:sampler-init-top-k %llama:sampler-init-top-p %llama:sampler-init-min-p
    %llama:sampler-init-dist
    ;; Embeddings
    %llama:model-n-embd %llama:get-embeddings-ith
    ;; Chat templates
    %llama:chat-apply-template %llama:chat-builtin-templates
    %llama:chat-message %llama:model-chat-template
    ;; LoRA adapters
    %llama:adapter-lora-init %llama:adapter-lora-free
    %llama:set-adapters-lora
    %llama:adapter-meta-val-str %llama:adapter-meta-count
    %llama:adapter-meta-key-by-index %llama:adapter-meta-val-str-by-index
    %llama:adapter-get-alora-n-invocation-tokens
    %llama:adapter-get-alora-invocation-tokens
    ;; Control vectors
    %llama:set-adapter-cvec
    ;; KV cache / memory management
    %llama:memory-seq-rm %llama:memory-seq-cp %llama:memory-seq-keep
    %llama:memory-seq-add %llama:memory-seq-div
    %llama:memory-seq-pos-min %llama:memory-seq-pos-max
    %llama:memory-can-shift
    ;; Extended sampler wrappers
    %llama:sampler-init-typical %llama:sampler-init-xtc
    %llama:sampler-init-top-n-sigma
    %llama:sampler-init-mirostat %llama:sampler-init-mirostat-v2
    %llama:sampler-init-temp-ext
    %llama:sampler-init-penalties %llama:sampler-init-dry
    %llama:sampler-init-logit-bias %llama:sampler-init-adaptive-p
    %llama:sampler-get-seed %llama:logit-bias
    %llama:vocab-n-tokens
    ;; Grammar / constrained generation
    %llama:sampler-init-grammar %llama:sampler-init-grammar-lazy
    %llama:sampler-init-grammar-lazy-patterns %llama:sampler-init-infill
    ;; Session state save/load
    %llama:state-get-size %llama:state-get-data %llama:state-set-data
    %llama:state-save-file %llama:state-load-file
    %llama:state-seq-get-size %llama:state-seq-get-data %llama:state-seq-set-data
    %llama:state-seq-save-file %llama:state-seq-load-file
    %llama:state-seq-get-size-ext %llama:state-seq-get-data-ext
    %llama:state-seq-set-data-ext
    ;; Model / context introspection
    %llama:model-desc %llama:model-size %llama:model-n-params
    %llama:model-n-ctx-train %llama:model-n-layer
    %llama:model-n-head %llama:model-n-head-kv
    %llama:model-n-embd-inp %llama:model-n-embd-out
    %llama:model-n-swa %llama:model-rope-type
    %llama:model-rope-freq-scale-train
    %llama:model-has-encoder %llama:model-has-decoder
    %llama:model-is-recurrent %llama:model-is-hybrid
    %llama:model-is-diffusion
    %llama:model-n-cls-out %llama:model-cls-label
    %llama:model-meta-count %llama:model-meta-key-by-index
    %llama:model-meta-val-str %llama:model-meta-val-str-by-index
    %llama:n-ctx %llama:n-batch %llama:n-ubatch %llama:n-seq-max
    %llama:n-threads %llama:n-threads-batch %llama:pooling-type
    %llama:print-system-info
    ;; Backend lifecycle
    %llama:backend-free %llama:numa-init
    ;; Context runtime configuration
    %llama:set-n-threads %llama:set-warmup %llama:set-causal-attn
    %llama:set-embeddings %llama:synchronize %llama:set-abort-callback
    ;; Threadpool management
    %llama:attach-threadpool %llama:detach-threadpool
    ;; Performance counters
    %llama:perf-context %llama:perf-context-print %llama:perf-context-reset
    %llama:perf-sampler %llama:perf-sampler-print %llama:perf-sampler-reset
    ;; Logging
    %llama:log-set %llama:log-get
    ;; System queries
    %llama:time-us %llama:max-devices
    %llama:supports-mmap %llama:supports-mlock
    %llama:supports-gpu-offload %llama:supports-rpc
    ;; Backend device introspection
    %llama:ggml-backend-dev-count %llama:ggml-backend-dev-get
    %llama:ggml-backend-dev-name %llama:ggml-backend-dev-description
    %llama:ggml-backend-dev-type %llama:ggml-backend-dev-memory
    %llama:ggml-backend-dev-get-props
    %llama:ggml-backend-dev-by-name %llama:ggml-backend-dev-by-type
    ;; Backend registry introspection
    %llama:ggml-backend-reg-count %llama:ggml-backend-reg-get
    %llama:ggml-backend-reg-name
    %llama:ggml-backend-reg-dev-count %llama:ggml-backend-reg-dev-get
    %llama:ggml-backend-reg-by-name
    ;; GGUF file inspection
    %llama:gguf-init-from-file %llama:gguf-free
    %llama:gguf-type-name
    %llama:gguf-get-version %llama:gguf-get-alignment %llama:gguf-get-data-offset
    %llama:gguf-get-n-kv %llama:gguf-find-key %llama:gguf-get-key
    %llama:gguf-get-kv-type %llama:gguf-get-arr-type
    %llama:gguf-get-val-u8 %llama:gguf-get-val-i8
    %llama:gguf-get-val-u16 %llama:gguf-get-val-i16
    %llama:gguf-get-val-u32 %llama:gguf-get-val-i32
    %llama:gguf-get-val-f32
    %llama:gguf-get-val-u64 %llama:gguf-get-val-i64 %llama:gguf-get-val-f64
    %llama:gguf-get-val-bool %llama:gguf-get-val-str %llama:gguf-get-val-data
    %llama:gguf-get-arr-n %llama:gguf-get-arr-data %llama:gguf-get-arr-str
    %llama:gguf-get-n-tensors %llama:gguf-find-tensor
    %llama:gguf-get-tensor-name %llama:gguf-get-tensor-type
    %llama:gguf-get-tensor-offset %llama:gguf-get-tensor-size))

(defun check-binding-deps ()
  "Verify that every symbol in *BINDING-DEPS* is fbound or a known type.
Returns T if all present, signals a warning per missing symbol."
  (let ((missing nil))
    (dolist (sym *binding-deps*)
      (unless (or (fboundp sym)
                  (ignore-errors (cffi:foreign-type-size sym))
                  (ignore-errors (cffi:foreign-type-size `(:struct ,sym))))
        (push sym missing)))
    (if missing
        (progn
          (warn "~D binding~:P missing from %llama after regeneration:~%~{  ~S~%~}"
                (length missing) (nreverse missing))
          nil)
        (progn
          (format t "~&All ~D binding dependencies present.~%" (length *binding-deps*))
          t))))
