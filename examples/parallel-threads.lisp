;;; Parallel contexts on separate threads — demonstrates the threading contract:
;;; independent LLAMA-CONTEXTs can run concurrently on separate Lisp threads,
;;; each generating text without interfering with the others.
;;;
;;; This uses SBCL's built-in sb-thread directly. Adapt to bordeaux-threads,
;;; lparallel, or any other concurrency library — cl-llama-cpp imposes no
;;; threading dependency.
;;;
;;; Threading rules illustrated here:
;;;   - One model loaded once; shared across threads (models are read-only).
;;;   - Each thread owns exactly one LLAMA-CONTEXT; never shared.
;;;   - ENSURE-BACKEND called once from the main thread before spawning workers.
;;;
;;; Setup:
;;;   export LLAMA_MODEL=/path/to/model.gguf
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/parallel-threads.lisp")
;;;   (cl-llama-cpp/examples/parallel-threads:run :n-threads 3)

(defpackage #:cl-llama-cpp/examples/parallel-threads
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/parallel-threads)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

#+sbcl
(defun run (&key (max-tokens 32) (temp 0.0) (seed 42))
  "Complete several prompts in parallel, one context per thread, then print results."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (let ((prompts '("The capital of France is"
                   "The speed of light is approximately"
                   "Water freezes at")))
    (format t "~&Loading model: ~A~2%" *model-path*)
    ;; Initialize the backend once on the main thread before spawning workers.
    ;; ensure-backend establishes a permanent hold so the worker threads'
    ;; with-context calls never race on backend-init/free.
    (ensure-backend)
    (with-model (model *model-path* :n-gpu-layers 99)
      (format t "Completing ~D prompts in parallel (one thread each):~%" (length prompts))
      (dolist (p prompts) (format t "  ~S~%" p))
      (terpri)
      (let* ((n (length prompts))
             (results (make-array n :initial-element nil))
             (threads
               (loop for prompt in prompts
                     for i from 0
                     collect
                     (let ((p prompt) (idx i))
                       (sb-thread:make-thread
                        (lambda ()
                          ;; Each thread owns its own context — never shared.
                          (with-context (ctx model :n-ctx 256)
                            (setf (aref results idx)
                                  (generate ctx p
                                            :max-tokens max-tokens
                                            :temp temp
                                            :seed seed))))
                        :name (format nil "llama-worker-~D" i))))))
        (mapc #'sb-thread:join-thread threads)
        (loop for prompt in prompts
              for i from 0
              do (format t "[~D] ~A~A~2%" (1+ i) prompt (aref results i))))))
  (values))

#-sbcl
(defun run (&rest args)
  (declare (ignore args))
  (error "This example uses sb-thread and requires SBCL. ~
          Adapt MAKE-THREAD / JOIN-THREAD calls to your implementation's ~
          threading API or to bordeaux-threads."))
