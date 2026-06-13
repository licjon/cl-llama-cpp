(in-package #:cl-llama-cpp)

(cffi:define-foreign-library libllama
  (:unix (:or "libllama.so.1" "libllama.so"))
  (:darwin (:or "libllama.1.dylib" "libllama.dylib"))
  (t (:default "libllama")))

(cffi:use-foreign-library libllama)

(defmacro with-fp-traps-masked (&body body)
  #+sbcl `(sb-int:with-float-traps-masked
              (:overflow :invalid :divide-by-zero :underflow :inexact)
            ,@body)
  #-sbcl `(progn ,@body))
