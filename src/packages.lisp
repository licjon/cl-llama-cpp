(defpackage #:%llama
  (:use))

(defpackage #:cl-llama-cpp
  (:use #:cl)
  (:documentation
   "High-level Common Lisp interface to llama.cpp.

Nullability convention — every high-level wrapper follows this rule:

  INPUT:  Lisp NIL  →  C null pointer   (for optional pointer arguments)
  OUTPUT: C null pointer  →  Lisp NIL   (for nullable return values)
          C null pointer  →  signal an error  (for allocation/init failures)

Users never need to call CFFI:NULL-POINTER or CFFI:NULL-POINTER-P.
New wrappers must follow this convention.")
  (:export
   ;; Typed handles
   #:llama-model #:llama-model-p #:llama-model-pointer
   #:llama-context #:llama-context-p #:llama-context-pointer
   #:llama-sampler #:llama-sampler-p #:llama-sampler-pointer
   #:ggml-backend-device #:ggml-backend-device-p #:ggml-backend-device-pointer
   #:ggml-backend-registry #:ggml-backend-registry-p #:ggml-backend-registry-pointer
   ;; Utility
   #:call-with-llama-compatible-fp-environment
   #:with-llama-compatible-fp-environment
   #:*binding-deps*
   #:check-binding-deps
   ;; Conditions
   #:llama-error
   #:model-load-error
   #:model-load-error-path
   #:context-creation-error
   #:tokenization-error
   #:tokenization-error-text
   #:decode-error
   #:decode-error-code
   ;; Resource management
   #:with-model
   #:with-context
   #:with-sampler-chain
   ;; Operations
   #:tokenize
   #:detokenize
   #:generate
   #:generate-parallel
   #:embed
   ;; Chat templates
   #:chat-template-error
   #:format-chat
   #:tokenize-chat
   #:model-chat-template
   #:list-chat-templates
   ;; LoRA adapters
   #:lora-load-error
   #:lora-load-error-path
   #:lora-apply-error
   #:lora-apply-error-code
   #:with-lora
   #:apply-lora
   #:lora-metadata
   ;; KV cache / memory management
   #:clear-kv-cache
   #:kv-cache-seq-rm
   #:kv-cache-seq-cp
   #:kv-cache-seq-keep
   #:kv-cache-seq-add
   #:kv-cache-seq-div
   #:kv-cache-pos
   #:kv-cache-can-shift-p
   ;; Session state save/load
   #:session-save-error
   #:session-save-error-path
   #:session-load-error
   #:session-load-error-path
   #:save-session
   #:load-session
   #:save-session-seq
   #:load-session-seq
   #:save-state
   #:load-state
   #:save-state-seq
   #:load-state-seq
   ;; Model / context introspection
   #:model-description
   #:model-metadata
   #:model-info
   #:model-cls-label
   #:context-info
   #:system-info
   ;; Grammar / constrained generation
   #:grammar-error
   #:grammar-error-grammar
   #:make-grammar-sampler
   #:make-grammar-sampler-lazy
   #:make-infill-sampler
   #:with-grammar-sampler
   ;; Sampler utilities
   #:sampler-seed
   ;; Batch API
   #:batch-init-error
   #:batch-init-error-n-tokens
   #:batch-overflow-error
   #:batch-overflow-error-capacity
   #:batch-overflow-error-token-count
   #:with-batch
   #:batch-add-token
   #:batch-add-embedding
   #:batch-add-sequence
   #:batch-clear
   #:batch-token-count
   #:batch-decode
   #:batch-encode
   ;; Backend lifecycle
   #:ensure-backend
   #:with-backend
   ;; Context runtime configuration
   #:set-n-threads
   #:set-warmup
   #:set-causal-attn
   #:set-embeddings
   #:synchronize
   #:set-abort-callback
   ;; Threadpool management
   #:attach-threadpool
   #:detach-threadpool
   ;; Performance counters
   #:context-perf
   #:print-context-perf
   #:reset-context-perf
   #:sampler-perf
   #:print-sampler-perf
   #:reset-sampler-perf
   #:print-perf
   #:reset-perf
   #:with-perf
   ;; Logging
   #:set-log-callback
   #:get-log-callback
   #:*last-log-callback-error*
   ;; System queries
   #:time-us
   #:system-capabilities
   ;; Backend device introspection
   #:backend-dev-count
   #:backend-dev-get
   #:backend-dev-name
   #:backend-dev-description
   #:backend-dev-type
   #:backend-dev-memory
   #:backend-dev-props
   #:backend-dev-by-name
   #:backend-dev-by-type
   ;; Backend registry introspection
   #:backend-reg-count
   #:backend-reg-get
   #:backend-reg-name
   #:backend-reg-dev-count
   #:backend-reg-dev-get
   #:backend-reg-by-name
   ;; High-level backend aggregates
   #:gpu-devices
   #:detect-free-vram
   #:detect-total-vram
   ;; Resource planning & configuration validation
   #:estimate-memory
   #:explain-memory-usage
   #:feasibility-report
   #:validate-configuration
   #:suggest-configuration
   #:configuration-unsafe-warning
   #:configuration-unsafe-error
   #:configuration-unsafe-error-reason))
