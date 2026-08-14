(defpackage #:cl-llama-cpp/common/reasoning-budget
  (:use #:cl)
  (:export
   ;; Conditions
   #:reasoning-budget-init-error
   #:reasoning-budget-init-error-message
   ;; State keywords
   #:+reasoning-budget-states+
   ;; Lifecycle
   #:make-reasoning-budget-sampler
   ;; Queries
   #:reasoning-budget-state
   #:reasoning-budget-force))
