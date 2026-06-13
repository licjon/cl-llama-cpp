(defpackage #:%llama
  (:use))

(defpackage #:cl-llama-cpp
  (:use #:cl)
  (:export #:with-fp-traps-masked))
