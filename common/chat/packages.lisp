(defpackage #:cl-llama-cpp/common/chat
  (:use #:cl)
  (:export
   ;; Conditions
   #:chat-init-error
   #:chat-init-error-message
   #:chat-parse-error
   #:chat-parse-error-message
   ;; Format constants
   #:+chat-format-content-only+
   #:+chat-format-peg-simple+
   #:+chat-format-peg-native+
   #:+chat-format-peg-gemma4+
   ;; Tool choice constants
   #:+tool-choice-auto+
   #:+tool-choice-required+
   #:+tool-choice-none+
   ;; Reasoning format constants
   #:+reasoning-format-none+
   #:+reasoning-format-auto+
   #:+reasoning-format-deepseek-legacy+
   #:+reasoning-format-deepseek+
   ;; Continuation constants
   #:+continuation-none+
   #:+continuation-auto+
   #:+continuation-reasoning+
   #:+continuation-content+
   ;; Lifecycle
   #:make-chat-templates
   #:free-chat-templates
   #:with-chat-templates
   ;; Queries
   #:chat-templates-source
   ;; Core API
   #:chat-templates-apply
   #:chat-parse
   #:chat-format-single
   #:chat-verify-template))
