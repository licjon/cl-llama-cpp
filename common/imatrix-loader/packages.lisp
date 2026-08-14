(defpackage #:cl-llama-cpp/common/imatrix-loader
  (:use #:cl)
  (:export
   ;; Conditions
   #:imatrix-load-error
   #:imatrix-load-error-filename
   ;; Data structures
   #:imatrix
   #:imatrix-entries
   #:imatrix-datasets
   #:imatrix-chunk-count
   #:imatrix-chunk-size
   #:imatrix-legacy-p
   #:imatrix-has-metadata-p
   #:imatrix-entry
   #:imatrix-entry-sums
   #:imatrix-entry-counts
   ;; API
   #:load-imatrix))
