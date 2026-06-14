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
   #:embed
   ;; Chat templates
   #:chat-template-error
   #:format-chat
   #:tokenize-chat
   #:model-chat-template
   #:list-chat-templates))
