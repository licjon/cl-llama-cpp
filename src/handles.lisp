(in-package #:cl-llama-cpp)

(defstruct (llama-model (:constructor %make-llama-model) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))

(defstruct (llama-context (:constructor %make-llama-context) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))

(defstruct (llama-sampler (:constructor %make-llama-sampler) (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))
