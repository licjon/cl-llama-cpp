(in-package #:cl-llama-cpp)

;;; Model / context introspection wrappers

(defun read-model-buffer-string (model reader-fn &rest extra-args)
  "Read a buffer-probe string from MODEL using READER-FN.
READER-FN is called as (apply reader-fn model-ptr ...extra-args buf buf-size)."
  (let ((model-ptr (llama-model-pointer model))
        (buf-size 256))
    (cffi:with-foreign-pointer (buf buf-size)
      (let ((n (apply reader-fn model-ptr (append extra-args (list buf buf-size)))))
        (when (< n 0)
          (return-from read-model-buffer-string nil))
        (when (>= n buf-size)
          (let ((retry-size (1+ n)))
            (cffi:with-foreign-pointer (buf2 retry-size)
              (let ((n2 (apply reader-fn model-ptr (append extra-args (list buf2 retry-size)))))
                (return-from read-model-buffer-string
                  (cffi:foreign-string-to-lisp buf2 :count (max 0 n2)))))))
        (cffi:foreign-string-to-lisp buf :count n)))))

(defun model-description (model)
  "Return MODEL's description as a string."
  (with-llama-compatible-fp-environment
    (read-model-buffer-string model #'%llama:model-desc)))

(defun model-metadata (model)
  "Return all metadata from MODEL as an alist of (key . value) string pairs."
  (with-llama-compatible-fp-environment
    (let ((count (%llama:model-meta-count (llama-model-pointer model))))
      (loop for i from 0 below count
            collect (cons
                     (read-model-buffer-string
                      model #'%llama:model-meta-key-by-index i)
                     (read-model-buffer-string
                      model #'%llama:model-meta-val-str-by-index i))))))

(defun model-info (model)
  "Return a plist of MODEL's numeric and boolean properties."
  (with-llama-compatible-fp-environment
    (let ((ptr (llama-model-pointer model)))
      (list :n-params (%llama:model-n-params ptr)
            :n-layers (%llama:model-n-layer ptr)
            :n-ctx-train (%llama:model-n-ctx-train ptr)
            :size-bytes (%llama:model-size ptr)
            :n-heads (%llama:model-n-head ptr)
            :n-heads-kv (%llama:model-n-head-kv ptr)
            :n-embd-in (%llama:model-n-embd-inp ptr)
            :n-embd-out (%llama:model-n-embd-out ptr)
            :n-swa (%llama:model-n-swa ptr)
            :rope-type (%llama:model-rope-type ptr)
            :rope-freq-scale (%llama:model-rope-freq-scale-train ptr)
            :n-cls-out (%llama:model-n-cls-out ptr)
            :encoder-p (not (zerop (%llama:model-has-encoder ptr)))
            :decoder-p (not (zerop (%llama:model-has-decoder ptr)))
            :recurrent-p (not (zerop (%llama:model-is-recurrent ptr)))
            :hybrid-p (not (zerop (%llama:model-is-hybrid ptr)))
            :diffusion-p (not (zerop (%llama:model-is-diffusion ptr)))))))

(defun model-cls-label (model index)
  "Return the classification label string at INDEX, or NIL if unavailable."
  (with-llama-compatible-fp-environment
    (%llama:model-cls-label (llama-model-pointer model) index)))

(defun context-info (ctx)
  "Return a plist of CTX's configuration properties."
  (with-llama-compatible-fp-environment
    (let ((ptr (llama-context-pointer ctx)))
      (list :n-ctx (%llama:n-ctx ptr)
            :n-batch (%llama:n-batch ptr)
            :n-ubatch (%llama:n-ubatch ptr)
            :n-seq-max (%llama:n-seq-max ptr)
            :n-threads (%llama:n-threads ptr)
            :n-threads-batch (%llama:n-threads-batch ptr)
            :pooling-type (%llama:pooling-type ptr)))))

(defun system-info ()
  "Return a string describing the llama.cpp build and system capabilities."
  (with-llama-compatible-fp-environment
    (%llama:print-system-info)))

;;; System queries

(defun time-us ()
  "Return the current wall-clock time in microseconds."
  (with-llama-compatible-fp-environment
    (%llama:time-us)))

(defun system-capabilities ()
  "Return a plist of system capability flags.
Keys: :MMAP :MLOCK :GPU-OFFLOAD :RPC :MAX-DEVICES"
  (with-llama-compatible-fp-environment
    (list :mmap        (not (zerop (%llama:supports-mmap)))
          :mlock       (not (zerop (%llama:supports-mlock)))
          :gpu-offload (not (zerop (%llama:supports-gpu-offload)))
          :rpc         (not (zerop (%llama:supports-rpc)))
          :max-devices (%llama:max-devices))))

;;; Performance counters

(defun context-perf (ctx)
  "Return performance data for CTX as a plist.
Keys: :T-START-MS :T-LOAD-MS :T-P-EVAL-MS :T-EVAL-MS :N-P-EVAL :N-EVAL :N-REUSED"
  (with-llama-compatible-fp-environment
    (let ((data (%llama:perf-context (llama-context-pointer ctx))))
      (list :t-start-ms  (getf data '%llama::t-start-ms)
            :t-load-ms   (getf data '%llama::t-load-ms)
            :t-p-eval-ms (getf data '%llama::t-p-eval-ms)
            :t-eval-ms   (getf data '%llama::t-eval-ms)
            :n-p-eval    (getf data '%llama::n-p-eval)
            :n-eval      (getf data '%llama::n-eval)
            :n-reused    (getf data '%llama::n-reused)))))

(defun print-context-perf (ctx)
  "Print context performance statistics for CTX to stderr."
  (with-llama-compatible-fp-environment
    (%llama:perf-context-print (llama-context-pointer ctx)))
  nil)

(defun reset-context-perf (ctx)
  "Reset context performance counters for CTX."
  (with-llama-compatible-fp-environment
    (%llama:perf-context-reset (llama-context-pointer ctx)))
  nil)

(defun sampler-perf (chain)
  "Return performance data for sampler CHAIN as a plist.
Keys: :T-SAMPLE-MS :N-SAMPLE"
  (with-llama-compatible-fp-environment
    (let ((data (%llama:perf-sampler (llama-sampler-pointer chain))))
      (list :t-sample-ms (getf data '%llama::t-sample-ms)
            :n-sample    (getf data '%llama::n-sample)))))

(defun print-sampler-perf (chain)
  "Print sampler performance statistics for CHAIN to stderr."
  (with-llama-compatible-fp-environment
    (%llama:perf-sampler-print (llama-sampler-pointer chain)))
  nil)

(defun reset-sampler-perf (chain)
  "Reset sampler performance counters for CHAIN."
  (with-llama-compatible-fp-environment
    (%llama:perf-sampler-reset (llama-sampler-pointer chain)))
  nil)

(defun print-perf (ctx)
  "Print context performance statistics for CTX to stderr."
  (print-context-perf ctx))

(defun reset-perf (ctx)
  "Reset context performance counters for CTX."
  (reset-context-perf ctx))

(defmacro with-perf ((ctx) &body body)
  "Reset CTX's performance counters, execute BODY, then print perf to stderr.
Performance is printed even on non-local exit. Returns the values of BODY."
  (let ((ctx-var (gensym "CTX")))
    `(let ((,ctx-var ,ctx))
       (reset-perf ,ctx-var)
       (unwind-protect
            (progn ,@body)
         (print-perf ,ctx-var)))))

;;; Logging

(defvar *log-callback* nil)
(defvar *last-log-callback-error* nil
  "The last error condition caught inside the log-callback panic boundary, or NIL.")

(defun log-error-to-safe-buffer (condition)
  (setf *last-log-callback-error* condition)
  (ignore-errors
    (format *debug-io* "~&[cl-llama-cpp] log callback error: ~A~%" condition)))

(cffi:defcallback %log-dispatcher :void
    ((level :int) (text :string) (data :pointer))
  (declare (ignore data))
  (when *log-callback*
    (handler-bind ((error (lambda (c)
                            (log-error-to-safe-buffer c)
                            (return-from %log-dispatcher))))
      (funcall *log-callback* level text))))

(defun set-log-callback (fn)
  "Set FN as the Lisp log callback for all llama.cpp log messages.
FN is called as (fn level text) where LEVEL is an integer (1=debug
2=info 3=warn 4=error) and TEXT is the message string.
Pass NIL to restore the default C stderr logger."
  (setf *log-callback* fn)
  (with-llama-compatible-fp-environment
    (%llama:log-set
     (if fn (cffi:callback %log-dispatcher) (cffi:null-pointer))
     (cffi:null-pointer)))
  nil)

(defun get-log-callback ()
  "Return the current Lisp log callback, or NIL if unset."
  *log-callback*)
