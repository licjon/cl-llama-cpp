(defpackage #:%llama
  (:use))

(defpackage #:cl-llama-cpp
  (:use #:cl)
  (:export
   ;; Utility
   #:with-fp-traps-masked
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
   #:batch-encode))
