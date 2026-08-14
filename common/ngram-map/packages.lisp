(defpackage #:cl-llama-cpp/common/ngram-map
  (:use #:cl)
  (:export
   ;; Conditions
   #:ngram-map-init-error
   #:ngram-map-init-error-message
   ;; Simple n-gram draft
   #:ngram-simple-draft
   ;; Map lifecycle
   #:make-ngram-map
   #:free-ngram-map
   #:with-ngram-map
   ;; Map operations
   #:ngram-map-begin
   #:ngram-map-draft
   #:ngram-map-accept))
