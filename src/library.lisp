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

(defvar *llama-fp-wrapper*
  #+sbcl
  (lambda (fn)
    (sb-int:with-float-traps-masked
        (:overflow :invalid :divide-by-zero :underflow :inexact)
      (funcall fn)))
  #-sbcl
  #'funcall)

(defun call-with-llama-compatible-fp-environment (fn)
  (funcall *llama-fp-wrapper* fn))

(defmacro with-llama-compatible-fp-environment (&body body)
  `(call-with-llama-compatible-fp-environment
     (lambda ()
       ,@body)))
