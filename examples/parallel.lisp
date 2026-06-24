;;; Parallel completions — demonstrates generate-parallel by completing
;;; multiple prompts simultaneously in shared forward passes.
;;;
;;; This is the batch API's core throughput use case: instead of processing
;;; prompts one at a time, all sequences share each decode call, making
;;; better use of GPU compute.
;;;
;;; Setup:
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/parallel.lisp")
;;;   (setf cl-llama-cpp/examples/parallel::*model-path*
;;;         "/path/to/model.gguf")
;;;   (cl-llama-cpp/examples/parallel:run)
;;;
;;; Or via environment variable:
;;;   export LLAMA_MODEL=/path/to/model.gguf

(defpackage #:cl-llama-cpp/examples/parallel
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/parallel)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

(defun run (&key (max-tokens 64) (temp 0.7))
  "Complete multiple prompts in parallel using the batch API."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (format t "~&Loading model: ~A~2%" *model-path*)
  (with-model (model *model-path* :n-gpu-layers 99)
    (with-context (ctx model :n-ctx 2048 :n-seq-max 8)
      (let ((prompts '("The secret to happiness is"
                        "In the year 2100, humans will"
                        "A recipe for disaster:")))
        (format t "Completing ~D prompts in parallel:~%" (length prompts))
        (dolist (p prompts)
          (format t "  ~S~%" p))
        (terpri)
        (multiple-value-bind (texts stop-reasons)
            (generate-parallel ctx prompts
                               :max-tokens max-tokens
                               :temp temp
                               :seed 42)
          (loop for prompt in prompts
                for text in texts
                for reason in stop-reasons
                for i from 1
                do (format t "[~D] ~A~A~%" i prompt text)
                   (format t "    (stopped: ~A)~2%" reason))))))
  (values))
