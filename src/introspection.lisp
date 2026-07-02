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

(llama-defun model-description (model)
  "Return MODEL's description as a string."
  (read-model-buffer-string model #'%llama:model-desc))

(llama-defun model-metadata (model)
  "Return all metadata from MODEL as an alist of (key . value) string pairs."
  (let ((count (%llama:model-meta-count (llama-model-pointer model))))
    (loop for i from 0 below count
          collect (cons
                   (read-model-buffer-string
                    model #'%llama:model-meta-key-by-index i)
                   (read-model-buffer-string
                    model #'%llama:model-meta-val-str-by-index i)))))

(llama-defun model-info (model)
  "Return a plist of MODEL's numeric and boolean properties."
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
          :diffusion-p (not (zerop (%llama:model-is-diffusion ptr))))))

(llama-defun model-cls-label (model index)
  "Return the classification label string at INDEX, or NIL if unavailable."
  (%llama:model-cls-label (llama-model-pointer model) index))

(llama-defun context-info (ctx)
  "Return a plist of CTX's configuration properties."
  (let ((ptr (llama-context-pointer ctx)))
    (list :n-ctx (%llama:n-ctx ptr)
          :n-batch (%llama:n-batch ptr)
          :n-ubatch (%llama:n-ubatch ptr)
          :n-seq-max (%llama:n-seq-max ptr)
          :n-threads (%llama:n-threads ptr)
          :n-threads-batch (%llama:n-threads-batch ptr)
          :pooling-type (%llama:pooling-type ptr))))

(llama-defun system-info ()
  "Return a string describing the llama.cpp build and system capabilities."
  (%llama:print-system-info))

;;; System queries

(llama-defun time-us ()
  "Return the current wall-clock time in microseconds."
  (%llama:time-us))

(llama-defun system-capabilities ()
  "Return a plist of system capability flags.
Keys: :MMAP :MLOCK :GPU-OFFLOAD :RPC :MAX-DEVICES
      :N-BACKEND-DEVS :N-BACKEND-REGS :HAS-GPU"
  (let* ((n-devs (%llama:ggml-backend-dev-count))
         (has-gpu (loop for i below n-devs
                        for dev-ptr = (%llama:ggml-backend-dev-get i)
                        for dev-type = (%llama:ggml-backend-dev-type dev-ptr)
                        thereis (member dev-type '(:gpu :igpu)))))
    (list :mmap           (not (zerop (%llama:supports-mmap)))
          :mlock          (not (zerop (%llama:supports-mlock)))
          :gpu-offload    (not (zerop (%llama:supports-gpu-offload)))
          :rpc            (not (zerop (%llama:supports-rpc)))
          :max-devices    (%llama:max-devices)
          :n-backend-devs n-devs
          :n-backend-regs (%llama:ggml-backend-reg-count)
          :has-gpu        (if has-gpu t nil))))

;;; Performance counters

(llama-defun context-perf (ctx)
  "Return performance data for CTX as a plist.
Keys: :T-START-MS :T-LOAD-MS :T-P-EVAL-MS :T-EVAL-MS :N-P-EVAL :N-EVAL :N-REUSED"
  (let ((data (%llama:perf-context (llama-context-pointer ctx))))
    (list :t-start-ms  (getf data '%llama::t-start-ms)
          :t-load-ms   (getf data '%llama::t-load-ms)
          :t-p-eval-ms (getf data '%llama::t-p-eval-ms)
          :t-eval-ms   (getf data '%llama::t-eval-ms)
          :n-p-eval    (getf data '%llama::n-p-eval)
          :n-eval      (getf data '%llama::n-eval)
          :n-reused    (getf data '%llama::n-reused))))

(llama-defun print-context-perf (ctx)
  "Print context performance statistics for CTX to stderr."
  (%llama:perf-context-print (llama-context-pointer ctx))
  nil)

(llama-defun reset-context-perf (ctx)
  "Reset context performance counters for CTX."
  (%llama:perf-context-reset (llama-context-pointer ctx))
  nil)

(llama-defun sampler-perf (chain)
  "Return performance data for sampler CHAIN as a plist.
Keys: :T-SAMPLE-MS :N-SAMPLE"
  (let ((data (%llama:perf-sampler (llama-sampler-pointer chain))))
    (list :t-sample-ms (getf data '%llama::t-sample-ms)
          :n-sample    (getf data '%llama::n-sample))))

(llama-defun print-sampler-perf (chain)
  "Print sampler performance statistics for CHAIN to stderr."
  (%llama:perf-sampler-print (llama-sampler-pointer chain))
  nil)

(llama-defun reset-sampler-perf (chain)
  "Reset sampler performance counters for CHAIN."
  (%llama:perf-sampler-reset (llama-sampler-pointer chain))
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

;;; Log callback lock (SBCL built-in; no-op on other implementations).
;;; Lock ordering rule: *LOG-LOCK* may be acquired while *BACKEND-LOCK* is
;;; held (backend-init logs during startup), but never the reverse.

#+sbcl
(defvar *log-lock* (sb-thread:make-mutex :name "cl-llama-cpp-log")
  "Mutex protecting *LOG-CALLBACK*.")

(defmacro %with-log-lock (&body body)
  #+sbcl `(sb-thread:with-mutex (*log-lock*) ,@body)
  #-sbcl `(progn ,@body))

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
  ;; Capture the callback under the lock, then call it without the lock to
  ;; avoid deadlock if the callback itself calls SET-LOG-CALLBACK.
  (let ((cb (%with-log-lock *log-callback*)))
    (when cb
      (handler-bind ((error (lambda (c)
                              (log-error-to-safe-buffer c)
                              (return-from %log-dispatcher))))
        (funcall cb level text)))))

(llama-defun set-log-callback (fn)
  "Set FN as the Lisp log callback for all llama.cpp log messages.
FN is called as (fn level text) where LEVEL is an integer (1=debug
2=info 3=warn 4=error) and TEXT is the message string.
Pass NIL to restore the default C stderr logger. Thread-safe."
  (%with-log-lock (setf *log-callback* fn))
  (%llama:log-set
   (if fn (cffi:callback %log-dispatcher) (cffi:null-pointer))
   (cffi:null-pointer))
  nil)

(defun get-log-callback ()
  "Return the current Lisp log callback, or NIL if unset."
  (%with-log-lock *log-callback*))

;;; Backend device introspection

(llama-defun backend-dev-count ()
  "Return the total number of registered backend devices."
  (%llama:ggml-backend-dev-count))

(llama-defun backend-dev-get (index)
  "Return the GGML-BACKEND-DEVICE at INDEX, or NIL if the pointer is null.
Signals an error if INDEX is out of range."
  (check-type index (integer 0))
  (let ((count (%llama:ggml-backend-dev-count)))
    (when (>= index count)
      (error "Device index ~D out of range [0, ~D)" index count))
    (let ((ptr (%llama:ggml-backend-dev-get index)))
      (unless (cffi:null-pointer-p ptr)
        (%make-ggml-backend-device :pointer ptr)))))

(llama-defun backend-dev-name (device)
  "Return the name string of DEVICE."
  (%llama:ggml-backend-dev-name (ggml-backend-device-pointer device)))

(llama-defun backend-dev-description (device)
  "Return the description string of DEVICE."
  (%llama:ggml-backend-dev-description (ggml-backend-device-pointer device)))

(llama-defun backend-dev-type (device)
  "Return the type of DEVICE as a keyword: :CPU :GPU :IGPU :ACCEL or :META."
  (%llama:ggml-backend-dev-type (ggml-backend-device-pointer device)))

(llama-defun backend-dev-memory (device)
  "Return free and total memory for DEVICE as (values free-bytes total-bytes)."
  (cffi:with-foreign-objects ((free-ptr '%llama::size-t)
                              (total-ptr '%llama::size-t))
    (%llama:ggml-backend-dev-memory (ggml-backend-device-pointer device)
                                    free-ptr total-ptr)
    (values (cffi:mem-ref free-ptr '%llama::size-t)
            (cffi:mem-ref total-ptr '%llama::size-t))))

(llama-defun backend-dev-props (device)
  "Return a plist of all properties for DEVICE.
Keys: :NAME :DESCRIPTION :MEMORY-FREE :MEMORY-TOTAL :TYPE :DEVICE-ID
      :ASYNC :HOST-BUFFER :BUFFER-FROM-HOST-PTR :EVENTS"
  (cffi:with-foreign-object (props '(:struct %llama::ggml-backend-dev-props))
    (%llama:ggml-backend-dev-get-props (ggml-backend-device-pointer device) props)
    (let ((caps-ptr (cffi:foreign-slot-pointer
                      props '(:struct %llama::ggml-backend-dev-props) '%llama::caps)))
      (list :name
            (cffi:foreign-slot-value props '(:struct %llama::ggml-backend-dev-props) '%llama::name)
            :description
            (cffi:foreign-slot-value props '(:struct %llama::ggml-backend-dev-props) '%llama::description)
            :memory-free
            (cffi:foreign-slot-value props '(:struct %llama::ggml-backend-dev-props) '%llama::memory-free)
            :memory-total
            (cffi:foreign-slot-value props '(:struct %llama::ggml-backend-dev-props) '%llama::memory-total)
            :type
            (cffi:foreign-slot-value props '(:struct %llama::ggml-backend-dev-props) '%llama::type)
            :device-id
            (cffi:foreign-slot-value props '(:struct %llama::ggml-backend-dev-props) '%llama::device-id)
            :async
            (not (zerop (cffi:foreign-slot-value caps-ptr '(:struct %llama::ggml-backend-dev-caps) '%llama::async)))
            :host-buffer
            (not (zerop (cffi:foreign-slot-value caps-ptr '(:struct %llama::ggml-backend-dev-caps) '%llama::host-buffer)))
            :buffer-from-host-ptr
            (not (zerop (cffi:foreign-slot-value caps-ptr '(:struct %llama::ggml-backend-dev-caps) '%llama::buffer-from-host-ptr)))
            :events
            (not (zerop (cffi:foreign-slot-value caps-ptr '(:struct %llama::ggml-backend-dev-caps) '%llama::events)))))))

(llama-defun backend-dev-by-name (name)
  "Return the GGML-BACKEND-DEVICE named NAME, or NIL if not found."
  (let ((ptr (%llama:ggml-backend-dev-by-name name)))
    (unless (cffi:null-pointer-p ptr)
      (%make-ggml-backend-device :pointer ptr))))

(llama-defun backend-dev-by-type (type)
  "Return the first GGML-BACKEND-DEVICE of TYPE (a keyword), or NIL if not found.
TYPE is one of :CPU :GPU :IGPU :ACCEL :META."
  (let ((ptr (%llama:ggml-backend-dev-by-type type)))
    (unless (cffi:null-pointer-p ptr)
      (%make-ggml-backend-device :pointer ptr))))

;;; Backend registry introspection

(llama-defun backend-reg-count ()
  "Return the total number of registered backend registries."
  (%llama:ggml-backend-reg-count))

(llama-defun backend-reg-get (index)
  "Return the GGML-BACKEND-REGISTRY at INDEX, or NIL if the pointer is null.
Signals an error if INDEX is out of range."
  (check-type index (integer 0))
  (let ((count (%llama:ggml-backend-reg-count)))
    (when (>= index count)
      (error "Registry index ~D out of range [0, ~D)" index count))
    (let ((ptr (%llama:ggml-backend-reg-get index)))
      (unless (cffi:null-pointer-p ptr)
        (%make-ggml-backend-registry :pointer ptr)))))

(llama-defun backend-reg-name (reg)
  "Return the name string of registry REG."
  (%llama:ggml-backend-reg-name (ggml-backend-registry-pointer reg)))

(llama-defun backend-reg-dev-count (reg)
  "Return the number of devices in registry REG."
  (%llama:ggml-backend-reg-dev-count (ggml-backend-registry-pointer reg)))

(llama-defun backend-reg-dev-get (reg index)
  "Return the GGML-BACKEND-DEVICE at INDEX within registry REG, or NIL if null.
Signals an error if INDEX is out of range."
  (check-type index (integer 0))
  (let ((count (%llama:ggml-backend-reg-dev-count (ggml-backend-registry-pointer reg))))
    (when (>= index count)
      (error "Registry device index ~D out of range [0, ~D)" index count))
    (let ((ptr (%llama:ggml-backend-reg-dev-get (ggml-backend-registry-pointer reg) index)))
      (unless (cffi:null-pointer-p ptr)
        (%make-ggml-backend-device :pointer ptr)))))

(llama-defun backend-reg-by-name (name)
  "Return the GGML-BACKEND-REGISTRY named NAME, or NIL if not found."
  (let ((ptr (%llama:ggml-backend-reg-by-name name)))
    (unless (cffi:null-pointer-p ptr)
      (%make-ggml-backend-registry :pointer ptr))))

;;; High-level backend aggregates

(llama-defun gpu-devices ()
  "Return a list of property plists for all registered GPU and IGPU devices.
Each plist has the keys from BACKEND-DEV-PROPS."
  (let ((count (%llama:ggml-backend-dev-count))
        (result nil))
    (dotimes (i count (nreverse result))
      (let ((ptr (%llama:ggml-backend-dev-get i)))
        (when (member (%llama:ggml-backend-dev-type ptr) '(:gpu :igpu))
          (push (backend-dev-props (%make-ggml-backend-device :pointer ptr))
                result))))))

(defun detect-free-vram ()
  "Return total free VRAM in bytes across all GPU and IGPU devices.
Returns NIL if no GPU devices are registered."
  (let ((devices (gpu-devices)))
    (when devices
      (reduce #'+ devices :key (lambda (p) (getf p :memory-free))))))

(defun detect-total-vram ()
  "Return total VRAM capacity in bytes across all GPU and IGPU devices.
Returns NIL if no GPU devices are registered."
  (let ((devices (gpu-devices)))
    (when devices
      (reduce #'+ devices :key (lambda (p) (getf p :memory-total))))))
