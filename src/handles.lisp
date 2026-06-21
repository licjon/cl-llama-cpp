(in-package #:cl-llama-cpp)

(defstruct (llama-model (:constructor %make-llama-model) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer)
  (freed-cell (list nil) :type cons :read-only t))

(defstruct (llama-context (:constructor %make-llama-context) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer)
  (compute-pending-p nil :type boolean)
  (freed-cell (list nil) :type cons :read-only t))

(defstruct (llama-sampler (:constructor %make-llama-sampler) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))

(defstruct (ggml-backend-device (:constructor %make-ggml-backend-device) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))

(defstruct (ggml-backend-registry (:constructor %make-ggml-backend-registry) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))

(defstruct (gguf-context (:constructor %make-gguf-context) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))
