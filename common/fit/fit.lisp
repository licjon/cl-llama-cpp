(in-package #:cl-llama-cpp/common/fit)

;;; Conditions

(define-condition fit-error (cl-llama-cpp:llama-error)
  ((path :initarg :path :reader fit-error-path))
  (:report (lambda (c s)
             (format s "Fit failed for model ~A" (fit-error-path c)))))

;;; Data structures

(defstruct (fit-result (:constructor %make-fit-result))
  (status :error :type keyword :read-only t)
  (n-gpu-layers 0 :type (signed-byte 32) :read-only t)
  (n-ctx 0 :type (unsigned-byte 32) :read-only t)
  (tensor-split #() :type simple-vector :read-only t))

;;; Status mapping

(defun %status-keyword (code)
  (case code
    (0 :success)
    (1 :failure)
    (t :error)))

;;; Unmarshaling

(defun %read-tensor-split (ptr)
  (let* ((n (%fit-n-devices))
         (arr (make-array n :element-type 'single-float)))
    (dotimes (i n arr)
      (setf (aref arr i) (%fit-tensor-split-at ptr i)))))

;;; Public API

(defun fit-params (path-model &key (margin 0) (n-ctx-min 512)
                                   (log-level :error))
  "Compute model/context parameters that fit available device memory.
PATH-MODEL is the path to a GGUF model file.
MARGIN is the bytes of memory to leave free per device (default 0).
N-CTX-MIN is the minimum context size when reducing to fit (default 512).
LOG-LEVEL is the llama.cpp log verbosity (:none :debug :info :warn :error).
Returns a FIT-RESULT struct with STATUS, N-GPU-LAYERS, N-CTX, and TENSOR-SPLIT.
Signals FIT-ERROR when a hard error occurs (e.g. model file not found).
NOTE: this function is NOT thread safe."
  (cl-llama-cpp:ensure-backend)
  (let* ((abs-path (namestring (merge-pathnames path-model)))
         (log-int (cffi:foreign-enum-value '%llama:ggml-log-level log-level))
         (ptr (%fit-create)))
    (when (cffi:null-pointer-p ptr)
      (error 'fit-error :path abs-path))
    (unwind-protect
         (let ((status-code (%fit-run ptr abs-path margin n-ctx-min log-int)))
           (when (= status-code 2)
             (error 'fit-error :path abs-path))
           (%make-fit-result
            :status (%status-keyword status-code)
            :n-gpu-layers (%fit-n-gpu-layers ptr)
            :n-ctx (%fit-n-ctx ptr)
            :tensor-split (%read-tensor-split ptr)))
      (%fit-free ptr))))

(defun fit-print (path-model)
  "Print estimated per-device memory breakdown for a model to stdout.
PATH-MODEL is the path to a GGUF model file.
Signals FIT-ERROR if the model cannot be loaded."
  (cl-llama-cpp:ensure-backend)
  (let* ((abs-path (namestring (merge-pathnames path-model)))
         (result (%fit-print abs-path)))
    (unless (zerop result)
      (error 'fit-error :path abs-path)))
  nil)

(defun memory-breakdown-print (context)
  "Print per-device memory breakdown for a loaded context to stdout.
CONTEXT is a CL-LLAMA-CPP:LLAMA-CONTEXT handle."
  (%fit-memory-breakdown-print (cl-llama-cpp:llama-context-pointer context))
  nil)
