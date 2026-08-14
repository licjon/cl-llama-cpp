(defpackage #:cl-llama-cpp/common/speculative
  (:use #:cl)
  (:export
   ;; Conditions
   #:speculative-init-error
   #:speculative-init-error-message
   ;; Type constants
   #:+speculative-type-none+
   #:+speculative-type-draft-simple+
   #:+speculative-type-draft-eagle3+
   #:+speculative-type-draft-mtp+
   #:+speculative-type-ngram-simple+
   #:+speculative-type-ngram-map-k+
   #:+speculative-type-ngram-map-k4v+
   #:+speculative-type-ngram-mod+
   #:+speculative-type-ngram-cache+
   ;; Params builder
   #:make-speculative-params
   #:free-speculative-params
   #:with-speculative-params
   #:speculative-params-add-type
   #:speculative-params-set-draft-n-max
   #:speculative-params-set-draft-n-min
   #:speculative-params-set-draft-p-min
   #:speculative-params-set-draft-p-split
   #:speculative-params-set-ngram-n
   #:speculative-params-set-ngram-m
   #:speculative-params-set-ngram-min-hits
   #:speculative-params-n-max
   ;; Lifecycle
   #:make-speculative-context
   #:free-speculative-context
   #:with-speculative-context
   ;; Operations
   #:speculative-begin
   #:speculative-draft
   #:speculative-accept
   #:speculative-need-embd-p
   #:speculative-need-embd-nextn-p
   #:speculative-print-stats))
