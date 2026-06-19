(in-package #:cl-llama-cpp)

(defvar *backend-initialized* nil)

(defun ensure-backend ()
  (unless *backend-initialized*
    (with-llama-compatible-fp-environment
      (%llama:backend-init))
    (setf *backend-initialized* t)))

(defmacro with-backend ((&key numa) &body body)
  "Initialize the llama backend, execute BODY, then shut it down.
On non-local exit the backend is always freed. Nesting is safe: only the
outermost WITH-BACKEND shuts down the backend. When :NUMA is a
ggml-numa-strategy keyword (:distribute :isolate :numactl :mirror),
calls llama-numa-init before executing BODY."
  (let ((outermost (gensym "OUTERMOST")))
    `(let ((,outermost (not *backend-initialized*)))
       (unless *backend-initialized*
         (with-llama-compatible-fp-environment (%llama:backend-init))
         (setf *backend-initialized* t))
       ,(when numa
          `(with-llama-compatible-fp-environment (%llama:numa-init ,numa)))
       (unwind-protect
            (progn ,@body)
         (when ,outermost
           (with-llama-compatible-fp-environment (%llama:backend-free))
           (setf *backend-initialized* nil))))))

(defun %bool->c (x) (if x 1 0))

(defun override-params (defaults overrides)
  "Override struct plist DEFAULTS with keyword OVERRIDES.
OVERRIDES is a plist like (:n-gpu-layers 99). Keys are matched
against the default plist keys by symbol name."
  (let ((result (copy-list defaults)))
    (loop for (key val) on overrides by #'cddr
          do (let ((match (loop for (k v) on result by #'cddr
                                when (string= (symbol-name key) (symbol-name k))
                                return k)))
               (when match
                 (setf (getf result match) val))))
    result))

(defmacro with-model ((var path &rest params) &body body)
  "Load a model from PATH, bind it to VAR, execute BODY, free the model.
PARAMS are keyword overrides for llama_model_default_params (e.g. :n-gpu-layers 99)."
  (let ((model-ptr (gensym "MODEL"))
        (path-val (gensym "PATH")))
    `(progn
       (ensure-backend)
       (with-llama-compatible-fp-environment
         (let* ((,path-val ,path)
                (defaults (%llama:model-default-params))
                (model-params (override-params defaults (list ,@params)))
                (,model-ptr (%llama:model-load-from-file ,path-val model-params)))
           (when (cffi:null-pointer-p ,model-ptr)
             (error 'model-load-error :path ,path-val))
           (let ((,var ,model-ptr))
             (unwind-protect
                  (progn ,@body)
               (%llama:model-free ,var))))))))

(defmacro with-context ((var model &rest params) &body body)
  "Create an inference context from MODEL, bind to VAR, execute BODY, free context.
PARAMS are keyword overrides for llama_context_default_params (e.g. :n-ctx 2048).
Additional keywords :VALIDATION (:off :warn :error) and :VRAM-BUDGET (bytes)
control pre-creation resource validation."
  (let* ((validation-form (getf params :validation))
         (vram-budget-form (getf params :vram-budget))
         (clean-params (loop for (k v) on params by #'cddr
                             unless (member k '(:validation :vram-budget))
                             append (list k v)))
         (ctx-ptr (gensym "CTX"))
         (model-var (gensym "MODEL")))
    `(with-llama-compatible-fp-environment
       (let* ((,model-var ,model)
              (defaults (%llama:context-default-params))
              (ctx-params (override-params defaults (list ,@clean-params))))
         ,@(when validation-form
             `((%validate-context-params
                ,model-var ctx-params ,validation-form ,vram-budget-form)))
         (let ((,ctx-ptr (%llama:new-context-with-model ,model-var ctx-params)))
           (when (cffi:null-pointer-p ,ctx-ptr)
             (error 'context-creation-error))
           (let ((,var ,ctx-ptr))
             (unwind-protect
                  (progn ,@body)
               (%llama:free ,var))))))))

;;; Context runtime configuration

(defun set-n-threads (ctx n-threads n-threads-batch)
  "Set the number of threads on CTX for single-token decoding (N-THREADS)
and batch decoding (N-THREADS-BATCH). Both values are required together."
  (check-type n-threads (integer 0 *))
  (check-type n-threads-batch (integer 0 *))
  (with-llama-compatible-fp-environment
    (%llama:set-n-threads ctx n-threads n-threads-batch))
  nil)

(defun set-warmup (ctx warmup-p)
  "Enable or disable the warmup pass on CTX."
  (with-llama-compatible-fp-environment
    (%llama:set-warmup ctx (%bool->c warmup-p)))
  nil)

(defun set-causal-attn (ctx causal-attn-p)
  "Enable or disable causal attention on CTX."
  (with-llama-compatible-fp-environment
    (%llama:set-causal-attn ctx (%bool->c causal-attn-p)))
  nil)

(defun set-embeddings (ctx embeddings-p)
  "Set the embedding-output flag on CTX to EMBEDDINGS-P. Must match the mode
the context was created with: enable only on contexts created with :embeddings
non-nil, disable only on contexts created without it. Toggling on a mismatched
context leaves internal C state inconsistent."
  (with-llama-compatible-fp-environment
    (%llama:set-embeddings ctx (%bool->c embeddings-p)))
  nil)

(defun synchronize (ctx)
  "Block until all pending async operations on CTX have completed."
  (with-llama-compatible-fp-environment
    (%llama:synchronize ctx))
  nil)

(defun set-abort-callback (ctx callback &optional data)
  "Register an abort callback on CTX. CALLBACK must be a foreign function
pointer obtained via CFFI:CALLBACK, or NIL to clear the callback.
DATA is an optional opaque data pointer passed to the callback."
  (with-llama-compatible-fp-environment
    (%llama:set-abort-callback
     ctx
     (or callback (cffi:null-pointer))
     (or data (cffi:null-pointer))))
  nil)

;;; Threadpool management

(defun attach-threadpool (ctx threadpool &optional threadpool-batch)
  "Attach THREADPOOL to CTX. THREADPOOL-BATCH is an optional separate
threadpool for batch operations; when omitted, THREADPOOL is used for both.
The caller retains ownership of the threadpool — detach-threadpool does
not free it."
  (with-llama-compatible-fp-environment
    (%llama:attach-threadpool
     ctx
     threadpool
     (or threadpool-batch (cffi:null-pointer))))
  nil)

(defun detach-threadpool (ctx)
  "Detach the threadpool from CTX. Does not free the threadpool."
  (with-llama-compatible-fp-environment
    (%llama:detach-threadpool ctx))
  nil)
