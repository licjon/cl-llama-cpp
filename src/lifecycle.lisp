(in-package #:cl-llama-cpp)

;;; Threading contract for cl-llama-cpp
;;;
;;; llama.cpp threading guarantees (below the FFI line):
;;;   - Compute parallelism is handled inside ggml's threadpool, exposed via
;;;     SET-N-THREADS, ATTACH-THREADPOOL, and DETACH-THREADPOOL.
;;;   - A single LLAMA-CONTEXT must NOT be used from multiple Lisp threads
;;;     concurrently; there is no internal locking on DECODE.
;;;   - Independent contexts CAN run in parallel on separate Lisp threads.
;;;   - Parallel requests within one context use the batched sequence API
;;;     (GENERATE-PARALLEL) without requiring any Lisp threads.
;;;   - BACKEND-INIT/BACKEND-FREE and LOG-SET are process-global operations.
;;;
;;; What cl-llama-cpp guarantees (library-owned global state):
;;;   - ENSURE-BACKEND and WITH-BACKEND are internally thread-safe via a mutex
;;;     and reference count: concurrent calls neither double-initialize nor free
;;;     the backend while another scope is active.
;;;   - SET-LOG-CALLBACK and the internal log dispatcher are thread-safe.
;;;   - Every C entry point is wrapped in WITH-LLAMA-COMPATIBLE-FP-ENVIRONMENT
;;;     on the calling Lisp thread; FP trap masking is per-thread.
;;;   - No Lisp concurrency library (bordeaux-threads, lparallel, etc.) is
;;;     imported or required; users supply their own threading tool.

;;; --- Backend lifecycle lock (SBCL built-in; no-op on other implementations) --

#+sbcl
(defvar *backend-lock* (sb-thread:make-mutex :name "cl-llama-cpp-backend")
  "Mutex protecting backend lifecycle state variables.")

(defmacro %with-backend-lock (&body body)
  #+sbcl `(sb-thread:with-mutex (*backend-lock*) ,@body)
  #-sbcl `(progn ,@body))

;;; --- Backend state (all protected by *BACKEND-LOCK*) -------------------------

(defvar *backend-initialized* nil
  "T iff the llama backend is currently initialized.")

(defvar *backend-permanent* nil
  "T if ENSURE-BACKEND has been called. Prevents WITH-BACKEND from freeing
the backend when its scope depth reaches zero.")

(defvar *backend-refcount* 0
  "Count of active WITH-BACKEND scopes. Backend is freed when this reaches
zero, unless *BACKEND-PERMANENT* is T.")

;;; --- Internal scope helpers --------------------------------------------------

(defun %backend-scope-enter ()
  "Thread-safe entry into a WITH-BACKEND scope. Initializes the backend if not
already running, then increments the active scope count."
  (%with-backend-lock
    (unless *backend-initialized*
      (with-llama-compatible-fp-environment (%llama:backend-init))
      (setf *backend-initialized* t))
    (incf *backend-refcount*)))

(defun %backend-scope-exit ()
  "Thread-safe exit from a WITH-BACKEND scope. Frees the backend when the scope
count reaches zero and ENSURE-BACKEND has not established a permanent hold."
  (%with-backend-lock
    (when (zerop (decf *backend-refcount*))
      (unless *backend-permanent*
        (with-llama-compatible-fp-environment (%llama:backend-free))
        (setf *backend-initialized* nil)))))

;;; --- Public lifecycle API ---------------------------------------------------

(defun ensure-backend ()
  "Ensure the llama backend is initialized. Thread-safe; safe to call from any
thread. Establishes a permanent hold so the backend is not freed when an
enclosing WITH-BACKEND scope exits. Prefer calling this once from the main
thread before spawning worker threads."
  (%with-backend-lock
    (unless *backend-initialized*
      (with-llama-compatible-fp-environment (%llama:backend-init))
      (setf *backend-initialized* t))
    (setf *backend-permanent* t)))

(defmacro with-backend ((&key numa) &body body)
  "Initialize the llama backend, execute BODY, then shut it down on exit.
On non-local exit the backend is always freed. Nesting and concurrent calls
are both safe: only the first call to find the backend uninitialized initializes
it; the last active scope to exit frees it. If ENSURE-BACKEND was called first,
WITH-BACKEND will not free the backend on exit.
When :NUMA is a ggml-numa-strategy keyword (:distribute :isolate :numactl
:mirror), llama-numa-init is called after backend initialization."
  `(progn
     (%backend-scope-enter)
     ,@(when numa
         `((with-llama-compatible-fp-environment (%llama:numa-init ,numa))))
     (unwind-protect
          (progn ,@body)
       (%backend-scope-exit))))

(defun %bool->c (x) (if x 1 0))

(defun %coerce-bool-param (x)
  (etypecase x
    (integer x)
    (boolean (if x 1 0))))

(defparameter *bool-param-names*
  '("VOCAB-ONLY" "USE-MMAP" "USE-DIRECT-IO" "USE-MLOCK"
    "CHECK-TENSORS" "USE-EXTRA-BUFTS" "NO-HOST" "NO-ALLOC"
    "EMBEDDINGS" "OFFLOAD-KQV" "NO-PERF" "OP-OFFLOAD"
    "SWA-FULL" "KV-UNIFIED"))

(defun override-params (defaults overrides)
  "Override struct plist DEFAULTS with keyword OVERRIDES.
OVERRIDES is a plist like (:n-gpu-layers 99). Keys are matched
against the default plist keys by symbol name. Boolean-typed C
params (e.g. :embeddings, :vocab-only) accept T/NIL in addition
to 0/1."
  (let ((result (copy-list defaults)))
    (loop for (key val) on overrides by #'cddr
          do (let* ((key-name (symbol-name key))
                    (match (loop for (k v) on result by #'cddr
                                 when (string= key-name (symbol-name k))
                                 return k)))
               (when match
                 (setf (getf result match)
                       (if (member key-name *bool-param-names* :test #'string=)
                           (%coerce-bool-param val)
                           val)))))
    result))

;;; --- GC finalizer support (trivial-garbage) ---------------------------------

(defun %try-claim-for-free (cell)
  "Atomically transition (car CELL) from NIL to T. Returns T if this call won."
  #+sbcl (null (sb-ext:cas (car cell) nil t))
  #-sbcl (prog1 (null (car cell))
           (setf (car cell) t)))

(defun %register-model-finalizer (model)
  (let ((ptr (llama-model-pointer model))
        (cell (llama-model-freed-cell model)))
    (tg:finalize model
      (lambda ()
        (when (and (%try-claim-for-free cell)
                   *backend-initialized*)
          (ignore-errors
            (with-llama-compatible-fp-environment
              (%llama:model-free ptr))))))))

(defun %register-context-finalizer (ctx)
  (let ((ptr (llama-context-pointer ctx))
        (cell (llama-context-freed-cell ctx))
        (addr (cffi:pointer-address (llama-context-pointer ctx))))
    (tg:finalize ctx
      (lambda ()
        (when (and (%try-claim-for-free cell)
                   *backend-initialized*)
          (ignore-errors
            (with-llama-compatible-fp-environment
              (%llama:free ptr))
            (%with-abort-lock
              (remhash addr *abort-callbacks*))))))))

;;; --- Standalone model/context constructors & destructors --------------------

(defun make-model (path &rest params)
  "Load a model from PATH, return a LLAMA-MODEL handle with a GC finalizer.
PARAMS are keyword overrides for llama_model_default_params (e.g. :n-gpu-layers 99).
The caller owns the handle and must eventually call FREE-MODEL, or let the GC
collect it (the finalizer frees the foreign memory as a safety net)."
  (ensure-backend)
  (with-llama-compatible-fp-environment
    (let* ((defaults (%llama:model-default-params))
           (model-params (override-params defaults params))
           (ptr (%llama:model-load-from-file path model-params)))
      (when (cffi:null-pointer-p ptr)
        (error 'model-load-error :path path))
      (let ((handle (%make-llama-model :pointer ptr)))
        (%register-model-finalizer handle)
        handle))))

(defun free-model (model)
  "Free the foreign memory held by MODEL and cancel its GC finalizer.
Idempotent — calling on an already-freed model is a no-op. Returns NIL."
  (when (%try-claim-for-free (llama-model-freed-cell model))
    (tg:cancel-finalization model)
    (with-llama-compatible-fp-environment
      (%llama:model-free (llama-model-pointer model))))
  nil)

(defun make-context (model &rest params)
  "Create an inference context from MODEL, return a LLAMA-CONTEXT handle with
a GC finalizer. PARAMS are keyword overrides for llama_context_default_params
\(e.g. :n-ctx 2048). Additional keywords :VALIDATION (:off :warn :error) and
:VRAM-BUDGET (bytes) control pre-creation resource validation.
The caller owns the handle and must eventually call FREE-CONTEXT, or let the
GC collect it."
  (let* ((validation (getf params :validation))
         (vram-budget (getf params :vram-budget))
         (clean-params (loop for (k v) on params by #'cddr
                             unless (member k '(:validation :vram-budget))
                             collect k collect v)))
    (with-llama-compatible-fp-environment
      (let* ((defaults (%llama:context-default-params))
             (ctx-params (override-params defaults clean-params)))
        (when validation
          (%validate-context-params model ctx-params validation vram-budget))
        (let ((ptr (%llama:new-context-with-model
                    (llama-model-pointer model) ctx-params)))
          (when (cffi:null-pointer-p ptr)
            (error 'context-creation-error))
          (let ((handle (%make-llama-context :pointer ptr)))
            (%register-context-finalizer handle)
            handle))))))

(defun free-context (ctx)
  "Free the foreign memory held by CTX and cancel its GC finalizer.
Cleans up any registered abort callback. Idempotent. Returns NIL."
  (when (%try-claim-for-free (llama-context-freed-cell ctx))
    (tg:cancel-finalization ctx)
    (let ((ptr (llama-context-pointer ctx)))
      (with-llama-compatible-fp-environment
        (%llama:free ptr))
      (%with-abort-lock
        (remhash (cffi:pointer-address ptr) *abort-callbacks*))))
  nil)

;;; --- Scoped resource macros (now delegate to make/free) ---------------------

(defmacro with-model ((var path &rest params) &body body)
  "Load a model from PATH, bind it to VAR as a LLAMA-MODEL handle, execute BODY, free the model.
PARAMS are keyword overrides for llama_model_default_params (e.g. :n-gpu-layers 99)."
  `(let ((,var (make-model ,path ,@params)))
     (unwind-protect
          (progn ,@body)
       (free-model ,var))))

(defmacro with-context ((var model &rest params) &body body)
  "Create an inference context from MODEL, bind to VAR as a LLAMA-CONTEXT handle, execute BODY, free context.
PARAMS are keyword overrides for llama_context_default_params (e.g. :n-ctx 2048).
Additional keywords :VALIDATION (:off :warn :error) and :VRAM-BUDGET (bytes)
control pre-creation resource validation."
  `(let ((,var (make-context ,model ,@params)))
     (unwind-protect
          (progn ,@body)
       (free-context ,var))))

;;; Context runtime configuration

(defun set-n-threads (ctx n-threads n-threads-batch)
  "Set the number of threads on CTX for single-token decoding (N-THREADS)
and batch decoding (N-THREADS-BATCH). Both values are required together."
  (check-type n-threads (integer 0 *))
  (check-type n-threads-batch (integer 0 *))
  (with-llama-compatible-fp-environment
    (%llama:set-n-threads (llama-context-pointer ctx) n-threads n-threads-batch))
  nil)

(defun set-warmup (ctx warmup-p)
  "Enable or disable the warmup pass on CTX."
  (with-llama-compatible-fp-environment
    (%llama:set-warmup (llama-context-pointer ctx) (%bool->c warmup-p)))
  nil)

(defun set-causal-attn (ctx causal-attn-p)
  "Enable or disable causal attention on CTX."
  (with-llama-compatible-fp-environment
    (%llama:set-causal-attn (llama-context-pointer ctx) (%bool->c causal-attn-p)))
  nil)

(defun set-embeddings (ctx embeddings-p)
  "Set the embedding-output flag on CTX to EMBEDDINGS-P. Must match the mode
the context was created with: enable only on contexts created with :embeddings
non-nil, disable only on contexts created without it. Toggling on a mismatched
context leaves internal C state inconsistent."
  (with-llama-compatible-fp-environment
    (%llama:set-embeddings (llama-context-pointer ctx) (%bool->c embeddings-p)))
  nil)

(defun synchronize (ctx)
  "Block until all pending async operations on CTX have completed."
  (with-llama-compatible-fp-environment
    (%llama:synchronize (llama-context-pointer ctx)))
  (setf (llama-context-compute-pending-p ctx) nil)
  nil)

;;; --- Abort-callback safe dispatcher -------------------------------------------

#+sbcl
(defvar *abort-callback-lock* (sb-thread:make-mutex :name "cl-llama-cpp-abort")
  "Mutex protecting *ABORT-CALLBACKS*.")

(defmacro %with-abort-lock (&body body)
  #+sbcl `(sb-thread:with-mutex (*abort-callback-lock*) ,@body)
  #-sbcl `(progn ,@body))

(defvar *abort-callbacks* (make-hash-table)
  "Maps ctx-pointer addresses (integers) to registered Lisp abort callbacks.")

(defvar *last-abort-callback-error* nil
  "The last error condition caught inside the abort-callback panic boundary, or NIL.")

(defun %abort-error-to-safe-buffer (condition)
  (setf *last-abort-callback-error* condition)
  (ignore-errors
    (format *debug-io* "~&[cl-llama-cpp] abort callback error: ~A~%" condition)))

(cffi:defcallback %abort-dispatcher :bool ((data :pointer))
  ;; Called from a ggml worker thread. Never unwind past this boundary.
  ;; DATA is the ctx raw pointer, used as a key into *abort-callbacks*.
  (let ((fn (%with-abort-lock
              (gethash (cffi:pointer-address data) *abort-callbacks*))))
    (when fn
      (handler-bind ((error (lambda (c)
                              (%abort-error-to-safe-buffer c)
                              (return-from %abort-dispatcher nil))))
        (not (null (funcall fn)))))))

(defun set-abort-callback (ctx fn)
  "Register FN as the abort callback on CTX. FN is a Lisp function of no
arguments that returns T to abort generation or NIL to continue.
Pass NIL to clear a previously registered callback.

Thread-safety contract: FN runs on a ggml worker thread. Dynamic bindings
established on the calling thread are NOT visible inside FN. Do the bare
minimum — read an atomic flag, etc. — and use a thread-safe primitive to
communicate results back to the main thread. Never signal Lisp conditions
or do heavy work inside FN.

Errors inside FN are caught, stored in *LAST-ABORT-CALLBACK-ERROR*, and
generation continues (returns NIL); they never propagate across the FFI
boundary."
  (check-type fn (or null function))
  (let* ((ptr (llama-context-pointer ctx))
         (key (cffi:pointer-address ptr)))
    (%with-abort-lock
      (if fn
          (setf (gethash key *abort-callbacks*) fn)
          (remhash key *abort-callbacks*)))
    (with-llama-compatible-fp-environment
      (%llama:set-abort-callback
       ptr
       (if fn (cffi:callback %abort-dispatcher) (cffi:null-pointer))
       (if fn ptr (cffi:null-pointer)))))
  nil)

(defun get-abort-callback (ctx)
  "Return the Lisp abort callback registered on CTX, or NIL if none is set.
Thread-safe."
  (%with-abort-lock
    (gethash (cffi:pointer-address (llama-context-pointer ctx)) *abort-callbacks*)))

;;; Threadpool management

(defun attach-threadpool (ctx threadpool &optional threadpool-batch)
  "Attach THREADPOOL to CTX. THREADPOOL-BATCH is an optional separate
threadpool for batch operations; when omitted, THREADPOOL is used for both.
The caller retains ownership of the threadpool — detach-threadpool does
not free it."
  (with-llama-compatible-fp-environment
    (%llama:attach-threadpool
     (llama-context-pointer ctx)
     threadpool
     (or threadpool-batch (cffi:null-pointer))))
  nil)

(defun detach-threadpool (ctx)
  "Detach the threadpool from CTX. Does not free the threadpool."
  (with-llama-compatible-fp-environment
    (%llama:detach-threadpool (llama-context-pointer ctx)))
  nil)
