(in-package #:cl-llama-cpp)

(let ((submodule-lib (merge-pathnames "llama.cpp/build/bin/"
                                       (asdf:system-source-directory "cl-llama-cpp"))))
  (when (probe-file submodule-lib)
    (pushnew submodule-lib cffi:*foreign-library-directories* :test #'equal)))

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
