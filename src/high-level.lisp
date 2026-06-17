(in-package #:cl-llama-cpp)

;;; Every %llama symbol the high-level API depends on.  After regenerating
;;; bindings, call (cl-llama-cpp:check-binding-deps) to verify none were
;;; removed upstream.  The generator also reads this list to flag removals.
(defparameter *binding-deps*
  '(;; Lifecycle
    %llama:backend-init
    %llama:model-default-params %llama:model-load-from-file %llama:model-free
    %llama:context-default-params %llama:new-context-with-model %llama:free
    %llama:get-model %llama:get-memory %llama:memory-clear
    ;; Tokenization
    %llama:model-get-vocab %llama:tokenize %llama:detokenize %llama:token
    %llama:token-bos %llama:token-is-eog
    ;; Generation / batch
    %llama:batch-get-one %llama:batch-init %llama:batch-free
    %llama:decode %llama:encode
    %llama:sampler-chain-default-params %llama:sampler-chain-init
    %llama:sampler-chain-add %llama:sampler-sample %llama:sampler-accept
    %llama:sampler-free
    %llama:sampler-init-greedy %llama:sampler-init-temp
    %llama:sampler-init-top-k %llama:sampler-init-top-p %llama:sampler-init-min-p
    %llama:sampler-init-dist
    ;; Embeddings
    %llama:model-n-embd %llama:get-embeddings-ith
    ;; Chat templates
    %llama:chat-apply-template %llama:chat-builtin-templates
    %llama:chat-message %llama:model-chat-template
    ;; LoRA adapters
    %llama:adapter-lora-init %llama:adapter-lora-free
    %llama:set-adapters-lora
    %llama:adapter-meta-val-str %llama:adapter-meta-count
    %llama:adapter-meta-key-by-index %llama:adapter-meta-val-str-by-index
    %llama:adapter-get-alora-n-invocation-tokens
    %llama:adapter-get-alora-invocation-tokens
    ;; Control vectors
    %llama:set-adapter-cvec
    ;; KV cache / memory management
    %llama:memory-seq-rm %llama:memory-seq-cp %llama:memory-seq-keep
    %llama:memory-seq-add %llama:memory-seq-div
    %llama:memory-seq-pos-min %llama:memory-seq-pos-max
    %llama:memory-can-shift
    ;; Extended sampler wrappers
    %llama:sampler-init-typical %llama:sampler-init-xtc
    %llama:sampler-init-top-n-sigma
    %llama:sampler-init-mirostat %llama:sampler-init-mirostat-v2
    %llama:sampler-init-temp-ext
    %llama:sampler-init-penalties %llama:sampler-init-dry
    %llama:sampler-init-logit-bias %llama:sampler-init-adaptive-p
    %llama:sampler-get-seed %llama:logit-bias
    %llama:vocab-n-tokens
    ;; Grammar / constrained generation
    %llama:sampler-init-grammar %llama:sampler-init-grammar-lazy
    %llama:sampler-init-grammar-lazy-patterns %llama:sampler-init-infill
    ;; Session state save/load
    %llama:state-get-size %llama:state-get-data %llama:state-set-data
    %llama:state-save-file %llama:state-load-file
    %llama:state-seq-get-size %llama:state-seq-get-data %llama:state-seq-set-data
    %llama:state-seq-save-file %llama:state-seq-load-file
    %llama:state-seq-get-size-ext %llama:state-seq-get-data-ext
    %llama:state-seq-set-data-ext
    ;; Model / context introspection
    %llama:model-desc %llama:model-size %llama:model-n-params
    %llama:model-n-ctx-train %llama:model-n-layer
    %llama:model-n-head %llama:model-n-head-kv
    %llama:model-n-embd-inp %llama:model-n-embd-out
    %llama:model-n-swa %llama:model-rope-type
    %llama:model-rope-freq-scale-train
    %llama:model-has-encoder %llama:model-has-decoder
    %llama:model-is-recurrent %llama:model-is-hybrid
    %llama:model-is-diffusion
    %llama:model-n-cls-out %llama:model-cls-label
    %llama:model-meta-count %llama:model-meta-key-by-index
    %llama:model-meta-val-str %llama:model-meta-val-str-by-index
    %llama:n-ctx %llama:n-batch %llama:n-ubatch %llama:n-seq-max
    %llama:n-threads %llama:n-threads-batch %llama:pooling-type
    %llama:print-system-info
    ;; Backend lifecycle
    %llama:backend-free %llama:numa-init
    ;; Context runtime configuration
    %llama:set-n-threads %llama:set-warmup %llama:set-causal-attn
    %llama:set-embeddings %llama:synchronize %llama:set-abort-callback
    ;; Threadpool management
    %llama:attach-threadpool %llama:detach-threadpool
    ;; Performance counters
    %llama:perf-context %llama:perf-context-print %llama:perf-context-reset
    %llama:perf-sampler %llama:perf-sampler-print %llama:perf-sampler-reset
    ;; Logging
    %llama:log-set %llama:log-get
    ;; System queries
    %llama:time-us %llama:max-devices
    %llama:supports-mmap %llama:supports-mlock
    %llama:supports-gpu-offload %llama:supports-rpc))

(defun check-binding-deps ()
  "Verify that every symbol in *BINDING-DEPS* is fbound or a known type.
Returns T if all present, signals a warning per missing symbol."
  (let ((missing nil))
    (dolist (sym *binding-deps*)
      (unless (or (fboundp sym)
                  (ignore-errors (cffi:foreign-type-size sym))
                  (ignore-errors (cffi:foreign-type-size `(:struct ,sym))))
        (push sym missing)))
    (if missing
        (progn
          (warn "~D binding~:P missing from %llama after regeneration:~%~{  ~S~%~}"
                (length missing) (nreverse missing))
          nil)
        (progn
          (format t "~&All ~D binding dependencies present.~%" (length *binding-deps*))
          t))))

(defvar *backend-initialized* nil)

(defun ensure-backend ()
  (unless *backend-initialized*
    (with-fp-traps-masked
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
         (with-fp-traps-masked (%llama:backend-init))
         (setf *backend-initialized* t))
       ,(when numa
          `(with-fp-traps-masked (%llama:numa-init ,numa)))
       (unwind-protect
            (progn ,@body)
         (when ,outermost
           (with-fp-traps-masked (%llama:backend-free))
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
       (with-fp-traps-masked
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
PARAMS are keyword overrides for llama_context_default_params (e.g. :n-ctx 2048)."
  (let ((ctx-ptr (gensym "CTX")))
    `(with-fp-traps-masked
       (let* ((defaults (%llama:context-default-params))
              (ctx-params (override-params defaults (list ,@params)))
              (,ctx-ptr (%llama:new-context-with-model ,model ctx-params)))
         (when (cffi:null-pointer-p ,ctx-ptr)
           (error 'context-creation-error))
         (let ((,var ,ctx-ptr))
           (unwind-protect
                (progn ,@body)
             (%llama:free ,var)))))))

(defun tokenize (model text &key (add-special t) (parse-special nil))
  "Tokenize TEXT using MODEL's vocabulary. Returns a vector of token integers."
  (with-fp-traps-masked
    (let* ((vocab (%llama:model-get-vocab model))
           (text-len (length text))
           (add-sp (if add-special 1 0))
           (parse-sp (if parse-special 1 0))
           ;; First pass: get required token count
           (n-needed (- (%llama:tokenize vocab text text-len
                                         (cffi:null-pointer) 0
                                         add-sp parse-sp))))
      (when (<= n-needed 0)
        (error 'tokenization-error :text text))
      ;; Second pass: fill token buffer
      (cffi:with-foreign-object (buf '%llama:token n-needed)
        (let ((n-written (%llama:tokenize vocab text text-len
                                          buf n-needed
                                          add-sp parse-sp)))
          (when (< n-written 0)
            (error 'tokenization-error :text text))
          (let ((result (make-array n-written :element-type 'fixnum)))
            (dotimes (i n-written result)
              (setf (aref result i)
                    (cffi:mem-aref buf '%llama:token i)))))))))

(defun detokenize (model tokens &key (remove-special nil) (unparse-special t))
  "Detokenize a vector of TOKENS using MODEL's vocabulary. Returns a string."
  (with-fp-traps-masked
    (let* ((vocab (%llama:model-get-vocab model))
           (n-tokens (length tokens))
           (remove-sp (if remove-special 1 0))
           (unparse-sp (if unparse-special 1 0)))
      ;; Copy tokens into foreign buffer
      (cffi:with-foreign-object (tok-buf '%llama:token n-tokens)
        (dotimes (i n-tokens)
          (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
        ;; First pass: get required text length
        (let ((n-needed (- (%llama:detokenize vocab tok-buf n-tokens
                                              (cffi:null-pointer) 0
                                              remove-sp unparse-sp))))
          (when (<= n-needed 0)
            (return-from detokenize ""))
          ;; Second pass: fill text buffer (llama API does not null-terminate)
          (cffi:with-foreign-pointer (text-buf (1+ n-needed))
            (let ((n-written (%llama:detokenize vocab tok-buf n-tokens
                                                text-buf n-needed
                                                remove-sp unparse-sp)))
              (cffi:foreign-string-to-lisp text-buf :count n-written))))))))


(defun build-sampler-chain (&key (temp 0.8) top-k top-p min-p (seed 42) greedy
                                  model grammar (grammar-root "root") grammar-lazy
                                  grammar-trigger-words grammar-trigger-patterns
                                  grammar-trigger-tokens infill
                                  ;; Extended sampler keywords
                                  typical-p
                                  xtc-probability xtc-threshold
                                  top-n-sigma
                                  mirostat mirostat-v2
                                  (mirostat-tau 5.0) (mirostat-eta 0.1)
                                  repeat-penalty frequency-penalty
                                  presence-penalty (penalty-last-n 64)
                                  dry-multiplier (dry-base 1.75)
                                  (dry-allowed-length 2) (dry-penalty-last-n -1)
                                  dry-seq-breakers
                                  logit-bias
                                  dynamic-temp-range (dynamic-temp-exponent 1.0)
                                  adaptive-p (adaptive-p-decay 0.0))
  "Build and return a sampler chain pointer. Caller must free with %llama:sampler-free.
When GRAMMAR is provided, a grammar sampler is added (requires MODEL).
When INFILL is true, an infill sampler is added (requires MODEL).
When MIROSTAT or MIROSTAT-V2 is true, the normal top-k/top-p/min-p/temp/dist
chain is replaced with the mirostat sampler."
  (when (and mirostat mirostat-v2)
    (error "Cannot use both :MIROSTAT and :MIROSTAT-V2"))
  (when (and (or mirostat) (not model))
    (error ":MIROSTAT requires :MODEL"))
  (when (and dry-multiplier (not model))
    (error ":DRY-MULTIPLIER requires :MODEL"))
  (when (and logit-bias (not model))
    (error ":LOGIT-BIAS requires :MODEL"))
  (let ((chain (%llama:sampler-chain-init
                (%llama:sampler-chain-default-params))))
    ;; 1. Logit bias (modifies logits first)
    (when logit-bias
      (let* ((vocab (%llama:model-get-vocab model))
             (n-vocab (%llama:vocab-n-tokens vocab))
             (n-bias (length logit-bias)))
        (cffi:with-foreign-object (bias-buf '(:struct %llama:logit-bias) n-bias)
          (loop for (token-id . bias) in logit-bias
                for i from 0
                for ptr = (cffi:inc-pointer bias-buf (* i 8))
                do (setf (cffi:foreign-slot-value ptr '(:struct %llama:logit-bias) '%llama:token)
                         token-id)
                   (setf (cffi:foreign-slot-value ptr '(:struct %llama:logit-bias) '%llama:bias)
                         (coerce bias 'single-float)))
          (%llama:sampler-chain-add
           chain (%llama:sampler-init-logit-bias n-vocab n-bias bias-buf)))))
    ;; 2. Penalties (repeat/freq/presence)
    (when (or repeat-penalty frequency-penalty presence-penalty)
      (%llama:sampler-chain-add
       chain (%llama:sampler-init-penalties
              penalty-last-n
              (coerce (or repeat-penalty 1.0) 'single-float)
              (coerce (or frequency-penalty 0.0) 'single-float)
              (coerce (or presence-penalty 0.0) 'single-float))))
    ;; 3. DRY anti-repetition
    (when dry-multiplier
      (let* ((vocab (%llama:model-get-vocab model))
             (n-ctx-train (%llama:model-n-ctx-train model))
             (n-breakers (length dry-seq-breakers))
             (foreign-strings nil))
        (unwind-protect
            (cffi:with-foreign-object (str-buf :pointer (max 1 n-breakers))
              (loop for s in dry-seq-breakers
                    for i from 0
                    for fstr = (cffi:foreign-string-alloc s)
                    do (push fstr foreign-strings)
                       (setf (cffi:mem-aref str-buf :pointer i) fstr))
              (%llama:sampler-chain-add
               chain (%llama:sampler-init-dry
                      vocab n-ctx-train
                      (coerce dry-multiplier 'single-float)
                      (coerce dry-base 'single-float)
                      dry-allowed-length dry-penalty-last-n
                      (if (plusp n-breakers) str-buf (cffi:null-pointer))
                      n-breakers)))
          (dolist (ptr foreign-strings)
            (cffi:foreign-string-free ptr)))))
    ;; 4. Grammar / infill
    (when grammar
      (unless model (error ":GRAMMAR requires :MODEL"))
      (%llama:sampler-chain-add
       chain (if grammar-lazy
                 (make-grammar-sampler-lazy model grammar
                   :root grammar-root
                   :trigger-words grammar-trigger-words
                   :trigger-patterns grammar-trigger-patterns
                   :trigger-tokens grammar-trigger-tokens)
                 (make-grammar-sampler model grammar :root grammar-root))))
    (when infill
      (unless model (error ":INFILL requires :MODEL"))
      (%llama:sampler-chain-add chain (make-infill-sampler model)))
    ;; 5. Sampling strategy
    (cond
      (greedy
       (%llama:sampler-chain-add chain (%llama:sampler-init-greedy)))
      ((or mirostat mirostat-v2)
       (if mirostat
           (let* ((vocab (%llama:model-get-vocab model))
                  (n-vocab (%llama:vocab-n-tokens vocab)))
             (%llama:sampler-chain-add
              chain (%llama:sampler-init-mirostat
                     n-vocab seed
                     (coerce mirostat-tau 'single-float)
                     (coerce mirostat-eta 'single-float)
                     100)))
           (%llama:sampler-chain-add
            chain (%llama:sampler-init-mirostat-v2
                   seed
                   (coerce mirostat-tau 'single-float)
                   (coerce mirostat-eta 'single-float)))))
      (t
       (when top-k
         (%llama:sampler-chain-add chain (%llama:sampler-init-top-k top-k)))
       (when typical-p
         (%llama:sampler-chain-add
          chain (%llama:sampler-init-typical (coerce typical-p 'single-float) 1)))
       (when top-p
         (%llama:sampler-chain-add chain (%llama:sampler-init-top-p (coerce top-p 'single-float) 1)))
       (when min-p
         (%llama:sampler-chain-add chain (%llama:sampler-init-min-p (coerce min-p 'single-float) 1)))
       (when top-n-sigma
         (%llama:sampler-chain-add
          chain (%llama:sampler-init-top-n-sigma (coerce top-n-sigma 'single-float))))
       (when xtc-probability
         (%llama:sampler-chain-add
          chain (%llama:sampler-init-xtc
                 (coerce xtc-probability 'single-float)
                 (coerce (or xtc-threshold 0.5) 'single-float)
                 1 seed)))
       (when adaptive-p
         (%llama:sampler-chain-add
          chain (%llama:sampler-init-adaptive-p
                 (coerce adaptive-p 'single-float)
                 (coerce adaptive-p-decay 'single-float)
                 seed)))
       (if dynamic-temp-range
           (%llama:sampler-chain-add
            chain (%llama:sampler-init-temp-ext
                   (coerce temp 'single-float)
                   (coerce dynamic-temp-range 'single-float)
                   (coerce dynamic-temp-exponent 'single-float)))
           (%llama:sampler-chain-add
            chain (%llama:sampler-init-temp (coerce temp 'single-float))))
       (%llama:sampler-chain-add chain (%llama:sampler-init-dist seed))))
    chain))

(defmacro with-sampler-chain ((var &rest args
                                &key (temp 0.8) top-k top-p min-p
                                     (seed 42) greedy
                                     model grammar (grammar-root "root")
                                     grammar-lazy grammar-trigger-words
                                     grammar-trigger-patterns grammar-trigger-tokens
                                     infill
                                     typical-p xtc-probability xtc-threshold
                                     top-n-sigma mirostat mirostat-v2
                                     (mirostat-tau 5.0) (mirostat-eta 0.1)
                                     repeat-penalty frequency-penalty
                                     presence-penalty (penalty-last-n 64)
                                     dry-multiplier (dry-base 1.75)
                                     (dry-allowed-length 2) (dry-penalty-last-n -1)
                                     dry-seq-breakers logit-bias
                                     dynamic-temp-range (dynamic-temp-exponent 1.0)
                                     adaptive-p (adaptive-p-decay 0.0))
                              &body body)
  "Create a sampler chain, bind to VAR, execute BODY, free the chain."
  (declare (ignore temp top-k top-p min-p seed greedy
                   model grammar grammar-root grammar-lazy
                   grammar-trigger-words grammar-trigger-patterns
                   grammar-trigger-tokens infill
                   typical-p xtc-probability xtc-threshold
                   top-n-sigma mirostat mirostat-v2
                   mirostat-tau mirostat-eta
                   repeat-penalty frequency-penalty
                   presence-penalty penalty-last-n
                   dry-multiplier dry-base dry-allowed-length
                   dry-penalty-last-n dry-seq-breakers logit-bias
                   dynamic-temp-range dynamic-temp-exponent
                   adaptive-p adaptive-p-decay))
  (let ((chain (gensym "CHAIN")))
    `(with-fp-traps-masked
       (let ((,chain (build-sampler-chain ,@args)))
         (unwind-protect
              (let ((,var ,chain))
                ,@body)
           (%llama:sampler-free ,chain))))))

(defun generate (ctx prompt &key (max-tokens 256) (temp 0.8)
                                  top-k top-p min-p (seed 42)
                                  (parse-special t) prompt-tokens
                                  token-callback
                                  grammar (grammar-root "root")
                                  ;; Extended sampler keywords
                                  typical-p
                                  xtc-probability xtc-threshold
                                  top-n-sigma
                                  mirostat mirostat-v2
                                  (mirostat-tau 5.0) (mirostat-eta 0.1)
                                  repeat-penalty frequency-penalty
                                  presence-penalty (penalty-last-n 64)
                                  dry-multiplier (dry-base 1.75)
                                  (dry-allowed-length 2) (dry-penalty-last-n -1)
                                  dry-seq-breakers
                                  logit-bias
                                  dynamic-temp-range (dynamic-temp-exponent 1.0)
                                  adaptive-p (adaptive-p-decay 0.0))
  "Generate text by continuing PROMPT. Returns two values: the generated string
and a stop reason (:eog, :length, or :callback).
Uses the context's model for tokenization. Blocks until EOS or MAX-TOKENS.
Supports extended sampler keywords: :TYPICAL-P, :XTC-PROBABILITY, :XTC-THRESHOLD,
:MIROSTAT, :MIROSTAT-V2, :REPEAT-PENALTY, :FREQUENCY-PENALTY, :PRESENCE-PENALTY,
:DRY-MULTIPLIER, :LOGIT-BIAS, :TOP-N-SIGMA, :DYNAMIC-TEMP-RANGE, :ADAPTIVE-P, etc."
  (with-fp-traps-masked
    (let* ((model (%llama:get-model ctx))
           (vocab (%llama:model-get-vocab model))
           (prompt-tokens (or prompt-tokens
                              (tokenize model prompt :parse-special parse-special)))
           (n-prompt (length prompt-tokens))
           (generated (make-array 0 :element-type 'fixnum
                                    :adjustable t :fill-pointer 0)))
      (%llama:memory-clear (%llama:get-memory ctx) 1)
      ;; Decode the prompt
      (cffi:with-foreign-object (tok-buf '%llama:token n-prompt)
        (dotimes (i n-prompt)
          (setf (cffi:mem-aref tok-buf '%llama:token i) (aref prompt-tokens i)))
        (let* ((batch (%llama:batch-get-one tok-buf n-prompt))
               (rc (%llama:decode ctx batch)))
          (unless (zerop rc)
            (error 'decode-error :code rc))))
      ;; Generation loop
      (let ((sampler (build-sampler-chain
                      :temp temp :top-k top-k :top-p top-p
                      :min-p min-p :seed seed
                      :model model :grammar grammar
                      :grammar-root grammar-root
                      :typical-p typical-p
                      :xtc-probability xtc-probability
                      :xtc-threshold xtc-threshold
                      :top-n-sigma top-n-sigma
                      :mirostat mirostat :mirostat-v2 mirostat-v2
                      :mirostat-tau mirostat-tau :mirostat-eta mirostat-eta
                      :repeat-penalty repeat-penalty
                      :frequency-penalty frequency-penalty
                      :presence-penalty presence-penalty
                      :penalty-last-n penalty-last-n
                      :dry-multiplier dry-multiplier :dry-base dry-base
                      :dry-allowed-length dry-allowed-length
                      :dry-penalty-last-n dry-penalty-last-n
                      :dry-seq-breakers dry-seq-breakers
                      :logit-bias logit-bias
                      :dynamic-temp-range dynamic-temp-range
                      :dynamic-temp-exponent dynamic-temp-exponent
                      :adaptive-p adaptive-p
                      :adaptive-p-decay adaptive-p-decay))
            (emitted-len 0))
        (unwind-protect
            ;; sampler-sample already calls sampler-accept internally — do NOT
            ;; call sampler-accept again or the grammar FSM double-advances.
            (loop for i from 0 below max-tokens
                  for new-token = (%llama:sampler-sample sampler ctx -1)
                  until (not (zerop (%llama:token-is-eog vocab new-token)))
                  do (vector-push-extend new-token generated)
                     (when token-callback
                       (handler-case
                           (let* ((full (detokenize model generated
                                                    :remove-special t))
                                  (new-text (subseq full emitted-len)))
                             (when (plusp (length new-text))
                               (setf emitted-len (length full))
                               (unless (funcall token-callback new-text)
                                 (loop-finish))))
                         (error ())))
                     (cffi:with-foreign-object (tok-buf '%llama:token 1)
                       (setf (cffi:mem-aref tok-buf '%llama:token 0) new-token)
                       (let* ((batch (%llama:batch-get-one tok-buf 1))
                              (rc (%llama:decode ctx batch)))
                         (unless (zerop rc)
                           (error 'decode-error :code rc)))))
          (%llama:sampler-free sampler)))
      ;; Convert generated tokens to string
      (let ((text (if (zerop (length generated))
                      ""
                      (let ((result-tokens (make-array (length generated)
                                                       :element-type 'fixnum)))
                        (dotimes (i (length generated))
                          (setf (aref result-tokens i) (aref generated i)))
                        (detokenize model result-tokens :remove-special t))))
            (stop-reason (cond ((= (length generated) max-tokens) :length)
                               (t :eog))))
        (values text stop-reason)))))

(defun embed (ctx text &key (normalize t))
  "Compute embeddings for TEXT. Returns a vector of single-floats.
The context must have been created with :embeddings 1.
When NORMALIZE is true (default), L2-normalizes the result."
  (with-fp-traps-masked
    (let* ((model (%llama:get-model ctx))
           (tokens (tokenize model text))
           (n-tokens (length tokens))
           (n-embd (%llama:model-n-embd model)))
      ;; Build batch and encode
      (cffi:with-foreign-object (tok-buf '%llama:token n-tokens)
        (dotimes (i n-tokens)
          (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
        (let* ((batch (%llama:batch-get-one tok-buf n-tokens))
               (rc (%llama:encode ctx batch)))
          (unless (zerop rc)
            (error 'decode-error :code rc))))
      ;; Read embeddings
      (let* ((embd-ptr (%llama:get-embeddings-ith ctx 0))
             (result (make-array n-embd :element-type 'single-float)))
        (dotimes (i n-embd)
          (setf (aref result i) (cffi:mem-aref embd-ptr :float i)))
        (when normalize
          (let ((norm (sqrt (loop for x across result sum (* x x)))))
            (when (> norm 0.0)
              (dotimes (i (length result))
                (setf (aref result i) (/ (aref result i) norm))))))
        result))))

(defun model-chat-template (model &optional name)
  "Return the chat template string embedded in MODEL.
If NAME is given, look up a specific named template."
  (with-fp-traps-masked
    (%llama:model-chat-template model (or name (cffi:null-pointer)))))

(defun list-chat-templates ()
  "Return a list of built-in chat template name strings."
  (let ((n (%llama:chat-builtin-templates (cffi:null-pointer) 0)))
    (when (> n 0)
      (cffi:with-foreign-object (output :pointer n)
        (%llama:chat-builtin-templates output n)
        (loop for i below n
              collect (cffi:foreign-string-to-lisp
                       (cffi:mem-aref output :pointer i)))))))

(defun format-chat (model messages &key template (add-assistant-prefix t))
  "Format MESSAGES as a chat prompt string using a Jinja-style chat template.
MESSAGES is a list of plists with :role and :content keys.
Uses MODEL's embedded chat template unless TEMPLATE is provided."
  (with-fp-traps-masked
    (let* ((tmpl (or template (model-chat-template model)))
           (tmpl-arg (or tmpl (cffi:null-pointer)))
           ;; Gemma templates use "model" instead of "assistant"
           (model-role (and (stringp tmpl) (search "'model'" tmpl)))
           (n-msg (length messages))
           (add-ass (if add-assistant-prefix 1 0))
           (msg-size (cffi:foreign-type-size '(:struct %llama:chat-message)))
           (foreign-strings nil))
      (cffi:with-foreign-object (chat '(:struct %llama:chat-message) n-msg)
        (unwind-protect
            (progn
              (loop for msg in messages
                    for i from 0
                    for msg-ptr = (cffi:inc-pointer chat (* i msg-size))
                    for role = (let ((r (getf msg :role)))
                                 (if (and model-role (string= r "assistant"))
                                     "model" r))
                    for role-ptr = (cffi:foreign-string-alloc role)
                    for content-ptr = (cffi:foreign-string-alloc (getf msg :content))
                    do (push role-ptr foreign-strings)
                       (push content-ptr foreign-strings)
                       (setf (cffi:mem-ref msg-ptr :pointer 0) role-ptr
                             (cffi:mem-ref msg-ptr :pointer 8) content-ptr))
              (let ((n-needed (%llama:chat-apply-template
                               tmpl-arg chat n-msg add-ass
                               (cffi:null-pointer) 0)))
                (when (< n-needed 0)
                  (error 'chat-template-error))
                (cffi:with-foreign-pointer-as-string (buf (1+ n-needed))
                  (%llama:chat-apply-template
                   tmpl-arg chat n-msg add-ass
                   buf (1+ n-needed)))))
          (dolist (ptr foreign-strings)
            (cffi:foreign-string-free ptr)))))))

(defun tokenize-chat (model messages &key template (add-assistant-prefix t))
  "Tokenize a chat conversation safely. Template markers are parsed as special
tokens; message content is not. This prevents content that resembles special
tokens (e.g. a model hallucinating <end_of_turn>) from corrupting the prompt
on subsequent turns. Returns a token vector suitable for GENERATE's :prompt-tokens."
  (let* ((formatted (format-chat model messages
                      :template template
                      :add-assistant-prefix add-assistant-prefix))
         (vocab (%llama:model-get-vocab model))
         (bos (%llama:token-bos vocab))
         (all-tokens (make-array 0 :element-type 'fixnum
                                   :adjustable t :fill-pointer 0))
         (pos 0))
    ;; Explicitly prepend BOS and skip the template's bos_token rendering
    (vector-push-extend bos all-tokens)
    (let ((bos-text (detokenize model (make-array 1 :element-type 'fixnum
                                                    :initial-element bos))))
      (when (and (plusp (length bos-text))
                 (eql 0 (search bos-text formatted)))
        (setf pos (length bos-text))))
    (dolist (msg messages)
      (let* ((content (getf msg :content))
             (content-start (search content formatted :start2 pos)))
        (when content-start
          (when (> content-start pos)
            (let ((toks (tokenize model (subseq formatted pos content-start)
                          :add-special nil :parse-special t)))
              (loop for tok across toks do (vector-push-extend tok all-tokens))))
          (when (plusp (length content))
            (let ((toks (tokenize model content
                          :add-special nil :parse-special nil)))
              (loop for tok across toks do (vector-push-extend tok all-tokens))))
          (setf pos (+ content-start (length content))))))
    (when (< pos (length formatted))
      (let ((toks (tokenize model (subseq formatted pos)
                    :add-special nil :parse-special t)))
        (loop for tok across toks do (vector-push-extend tok all-tokens))))
    (let ((result (make-array (length all-tokens) :element-type 'fixnum)))
      (dotimes (i (length all-tokens))
        (setf (aref result i) (aref all-tokens i)))
      result)))

;;; LoRA adapter wrappers

(defmacro with-lora ((var model path) &body body)
  "Load a LoRA adapter from PATH for MODEL, bind it to VAR, execute BODY, free the adapter."
  (let ((adapter-ptr (gensym "ADAPTER"))
        (path-val (gensym "PATH")))
    `(progn
       (ensure-backend)
       (with-fp-traps-masked
         (let* ((,path-val ,path)
                (,adapter-ptr (%llama:adapter-lora-init ,model ,path-val)))
           (when (cffi:null-pointer-p ,adapter-ptr)
             (error 'lora-load-error :path ,path-val))
           (let ((,var ,adapter-ptr))
             (unwind-protect
                  (progn ,@body)
               (%llama:adapter-lora-free ,var))))))))

(defun apply-lora (ctx adapter &key (scale 1.0))
  "Set the active LoRA adapter on CTX to ADAPTER with the given SCALE factor.
Replaces any previously applied adapters — calling this twice does not
compose; only the last call's adapter remains active.
Returns NIL on success, signals LORA-APPLY-ERROR on failure."
  (with-fp-traps-masked
    (let ((scale-f (coerce scale 'single-float)))
      (unless (<= most-negative-single-float scale-f most-positive-single-float)
        (error 'type-error :datum scale :expected-type 'single-float))
      (cffi:with-foreign-objects ((adapters-buf :pointer 1)
                                  (scales-buf :float 1))
        (setf (cffi:mem-aref adapters-buf :pointer 0) adapter)
        (setf (cffi:mem-aref scales-buf :float 0) scale-f)
        (let ((rc (%llama:set-adapters-lora ctx adapters-buf 1 scales-buf)))
          (unless (zerop rc)
            (error 'lora-apply-error :code rc))
          nil)))))

(defun read-adapter-meta-string (adapter index reader-fn)
  "Read a metadata string from ADAPTER at INDEX using READER-FN.
READER-FN is called as (funcall reader-fn adapter index buf buf-size)."
  (let ((buf-size 256))
    (cffi:with-foreign-pointer (buf buf-size)
      (let ((n (funcall reader-fn adapter index buf buf-size)))
        (when (>= n buf-size)
          (let ((retry-size (1+ n)))
            (cffi:with-foreign-pointer (buf2 retry-size)
              (let ((n2 (funcall reader-fn adapter index buf2 retry-size)))
                (return-from read-adapter-meta-string
                  (cffi:foreign-string-to-lisp buf2 :count (max 0 n2)))))))
        (cffi:foreign-string-to-lisp buf :count (max 0 n))))))

(defun lora-metadata (adapter)
  "Return metadata from ADAPTER as an alist of (key . value) string pairs."
  (with-fp-traps-masked
    (let ((count (%llama:adapter-meta-count adapter)))
      (loop for i from 0 below count
            collect (cons
                     (read-adapter-meta-string
                      adapter i #'%llama:adapter-meta-key-by-index)
                     (read-adapter-meta-string
                      adapter i #'%llama:adapter-meta-val-str-by-index))))))

;;; KV cache / memory management wrappers

(defun clear-kv-cache (ctx)
  "Clear all KV cache state for CTX."
  (with-fp-traps-masked
    (%llama:memory-clear (%llama:get-memory ctx) 1))
  nil)

(defun kv-cache-seq-rm (ctx seq-id p0 p1)
  "Remove cached tokens in positions [P0, P1) for SEQ-ID.
P0=-1 means from the start, P1=-1 means to the end.
Returns T if cells were removed, NIL if no matching data."
  (with-fp-traps-masked
    (not (zerop (%llama:memory-seq-rm
                 (%llama:get-memory ctx) seq-id p0 p1)))))

(defun kv-cache-seq-cp (ctx src-seq dst-seq p0 p1)
  "Copy cached data from SRC-SEQ to DST-SEQ for positions [P0, P1)."
  (with-fp-traps-masked
    (%llama:memory-seq-cp (%llama:get-memory ctx) src-seq dst-seq p0 p1))
  nil)

(defun kv-cache-seq-keep (ctx seq-id)
  "Keep only SEQ-ID's cached data, removing all other sequences."
  (with-fp-traps-masked
    (%llama:memory-seq-keep (%llama:get-memory ctx) seq-id))
  nil)

(defun kv-cache-seq-add (ctx seq-id p0 p1 delta)
  "Shift positions in [P0, P1) for SEQ-ID by DELTA."
  (with-fp-traps-masked
    (%llama:memory-seq-add (%llama:get-memory ctx) seq-id p0 p1 delta))
  nil)

(defun kv-cache-seq-div (ctx seq-id p0 p1 d)
  "Divide positions in [P0, P1) for SEQ-ID by D. D must be non-zero."
  (when (zerop d)
    (error "Divisor must be non-zero"))
  (with-fp-traps-masked
    (%llama:memory-seq-div (%llama:get-memory ctx) seq-id p0 p1 d))
  nil)

(defun kv-cache-pos (ctx seq-id)
  "Return the minimum and maximum cached positions for SEQ-ID as (VALUES MIN MAX)."
  (with-fp-traps-masked
    (let ((mem (%llama:get-memory ctx)))
      (values (%llama:memory-seq-pos-min mem seq-id)
              (%llama:memory-seq-pos-max mem seq-id)))))

(defun kv-cache-can-shift-p (ctx)
  "Return T if CTX's memory supports position shifting, NIL otherwise."
  (with-fp-traps-masked
    (not (zerop (%llama:memory-can-shift (%llama:get-memory ctx))))))

;;; Session state save/load wrappers

(defun save-session (ctx path &optional tokens)
  "Save full context state to a session file at PATH.
TOKENS is an optional vector of token integers to store alongside the state."
  (with-fp-traps-masked
    (let* ((n-tokens (if tokens (length tokens) 0))
           (path-str (namestring path)))
      (if (zerop n-tokens)
          (let ((rc (%llama:state-save-file ctx path-str (cffi:null-pointer) 0)))
            (when (zerop rc)
              (error 'session-save-error :path path-str))
            nil)
          (let ((tok-buf (cffi:foreign-alloc '%llama:token :count n-tokens)))
            (unwind-protect
                (progn
                  (dotimes (i n-tokens)
                    (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
                  (let ((rc (%llama:state-save-file ctx path-str tok-buf n-tokens)))
                    (when (zerop rc)
                      (error 'session-save-error :path path-str))
                    nil))
              (cffi:foreign-free tok-buf)))))))

(defun load-session (ctx path)
  "Load context state from a session file at PATH.
Returns a vector of cached token integers that were stored with the state."
  (with-fp-traps-masked
    (let* ((path-str (namestring path))
           (capacity (%llama:n-ctx ctx))
           (tok-buf (cffi:foreign-alloc '%llama:token :count capacity)))
      (unwind-protect
          (cffi:with-foreign-object (count-out '%llama:size-t)
            (let ((rc (%llama:state-load-file ctx path-str tok-buf capacity count-out)))
              (when (zerop rc)
                (error 'session-load-error :path path-str))
              (let* ((n-tokens (cffi:mem-ref count-out '%llama:size-t))
                     (result (make-array n-tokens :element-type 'fixnum)))
                (dotimes (i n-tokens result)
                  (setf (aref result i)
                        (cffi:mem-aref tok-buf '%llama:token i))))))
        (cffi:foreign-free tok-buf)))))

(defun save-session-seq (ctx path seq-id &optional tokens)
  "Save a single sequence's state to a file at PATH.
TOKENS is an optional vector of token integers to store alongside the state."
  (with-fp-traps-masked
    (let* ((n-tokens (if tokens (length tokens) 0))
           (path-str (namestring path)))
      (if (zerop n-tokens)
          (let ((rc (%llama:state-seq-save-file
                     ctx path-str seq-id (cffi:null-pointer) 0)))
            (when (zerop rc)
              (error 'session-save-error :path path-str))
            nil)
          (let ((tok-buf (cffi:foreign-alloc '%llama:token :count n-tokens)))
            (unwind-protect
                (progn
                  (dotimes (i n-tokens)
                    (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
                  (let ((rc (%llama:state-seq-save-file
                             ctx path-str seq-id tok-buf n-tokens)))
                    (when (zerop rc)
                      (error 'session-save-error :path path-str))
                    nil))
              (cffi:foreign-free tok-buf)))))))

(defun load-session-seq (ctx path seq-id)
  "Load a single sequence's state from a file at PATH.
Returns a vector of cached token integers that were stored with the state."
  (with-fp-traps-masked
    (let* ((path-str (namestring path))
           (capacity (%llama:n-ctx ctx))
           (tok-buf (cffi:foreign-alloc '%llama:token :count capacity)))
      (unwind-protect
          (cffi:with-foreign-object (count-out '%llama:size-t)
            (let ((rc (%llama:state-seq-load-file
                       ctx path-str seq-id tok-buf capacity count-out)))
              (when (zerop rc)
                (error 'session-load-error :path path-str))
              (let* ((n-tokens (cffi:mem-ref count-out '%llama:size-t))
                     (result (make-array n-tokens :element-type 'fixnum)))
                (dotimes (i n-tokens result)
                  (setf (aref result i)
                        (cffi:mem-aref tok-buf '%llama:token i))))))
        (cffi:foreign-free tok-buf)))))

(defun save-state (ctx)
  "Serialize full context state to a Lisp octet vector."
  (with-fp-traps-masked
    (let ((size (%llama:state-get-size ctx)))
      (when (zerop size)
        (return-from save-state
          (make-array 0 :element-type '(unsigned-byte 8))))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (let* ((written (%llama:state-get-data ctx buf size))
                   (result (make-array written :element-type '(unsigned-byte 8))))
              (dotimes (i written result)
                (setf (aref result i) (cffi:mem-aref buf :uint8 i))))
          (cffi:foreign-free buf))))))

(defun load-state (ctx state-bytes)
  "Restore context state from a Lisp octet vector STATE-BYTES.
Returns the number of bytes consumed."
  (with-fp-traps-masked
    (let ((size (length state-bytes)))
      (when (zerop size)
        (return-from load-state 0))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (progn
              (dotimes (i size)
                (setf (cffi:mem-aref buf :uint8 i) (aref state-bytes i)))
              (%llama:state-set-data ctx buf size))
          (cffi:foreign-free buf))))))

(defun save-state-seq (ctx seq-id &key flags)
  "Serialize one sequence's state to a Lisp octet vector.
When FLAGS is provided, uses the extended variant with llama_state_seq_flags."
  (with-fp-traps-masked
    (let ((size (if flags
                    (%llama:state-seq-get-size-ext ctx seq-id flags)
                    (%llama:state-seq-get-size ctx seq-id))))
      (when (zerop size)
        (return-from save-state-seq
          (make-array 0 :element-type '(unsigned-byte 8))))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (let* ((written (if flags
                                (%llama:state-seq-get-data-ext ctx buf size seq-id flags)
                                (%llama:state-seq-get-data ctx buf size seq-id)))
                   (result (make-array written :element-type '(unsigned-byte 8))))
              (dotimes (i written result)
                (setf (aref result i) (cffi:mem-aref buf :uint8 i))))
          (cffi:foreign-free buf))))))

(defun load-state-seq (ctx seq-id state-bytes &key flags)
  "Restore one sequence's state from a Lisp octet vector STATE-BYTES.
When FLAGS is provided, uses the extended variant with llama_state_seq_flags.
Returns the number of bytes consumed."
  (with-fp-traps-masked
    (let ((size (length state-bytes)))
      (when (zerop size)
        (return-from load-state-seq 0))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (progn
              (dotimes (i size)
                (setf (cffi:mem-aref buf :uint8 i) (aref state-bytes i)))
              (if flags
                  (%llama:state-seq-set-data-ext ctx buf size seq-id flags)
                  (%llama:state-seq-set-data ctx buf size seq-id)))
          (cffi:foreign-free buf))))))

;;; Grammar / constrained generation wrappers

(defun make-grammar-sampler (model grammar &key (root "root"))
  "Create a grammar sampler from a GBNF grammar string and root rule.
Returns a sampler pointer. Caller must free with %llama:sampler-free,
or add to a sampler chain (which frees it automatically)."
  (check-type grammar string)
  (when (zerop (length grammar))
    (error 'grammar-error :grammar grammar))
  (with-fp-traps-masked
    (let* ((vocab (%llama:model-get-vocab model))
           (sampler (%llama:sampler-init-grammar vocab grammar root)))
      (when (cffi:null-pointer-p sampler)
        (error 'grammar-error :grammar grammar))
      sampler)))

(defun make-grammar-sampler-lazy (model grammar &key (root "root")
                                                      trigger-words
                                                      trigger-patterns
                                                      trigger-tokens)
  "Create a lazy grammar sampler that activates only when triggered.
When TRIGGER-PATTERNS is provided, uses pattern matching; otherwise uses
TRIGGER-WORDS for exact word matching. TRIGGER-TOKENS are token IDs that
also trigger grammar activation.
Returns a sampler pointer. Caller must free with %llama:sampler-free."
  (check-type grammar string)
  (when (zerop (length grammar))
    (error 'grammar-error :grammar grammar))
  (when (and trigger-words trigger-patterns)
    (error "Cannot specify both :TRIGGER-WORDS and :TRIGGER-PATTERNS"))
  (with-fp-traps-masked
    (let* ((vocab (%llama:model-get-vocab model))
           (use-patterns-p (not (null trigger-patterns)))
           (strings (if use-patterns-p trigger-patterns (or trigger-words nil)))
           (n-strings (length strings))
           (n-tokens (if trigger-tokens (length trigger-tokens) 0))
           (foreign-strings nil))
      (unwind-protect
          (cffi:with-foreign-objects ((str-buf :pointer (max 1 n-strings))
                                     (tok-buf '%llama:token (max 1 n-tokens)))
            (loop for s in strings
                  for i from 0
                  for fstr = (cffi:foreign-string-alloc s)
                  do (push fstr foreign-strings)
                     (setf (cffi:mem-aref str-buf :pointer i) fstr))
            (when trigger-tokens
              (dotimes (i n-tokens)
                (setf (cffi:mem-aref tok-buf '%llama:token i)
                      (elt trigger-tokens i))))
            (let* ((str-ptr (if (plusp n-strings) str-buf (cffi:null-pointer)))
                   (tok-ptr (if (plusp n-tokens) tok-buf (cffi:null-pointer)))
                   (sampler (if use-patterns-p
                                (%llama:sampler-init-grammar-lazy-patterns
                                 vocab grammar root
                                 str-ptr n-strings tok-ptr n-tokens)
                                (%llama:sampler-init-grammar-lazy
                                 vocab grammar root
                                 str-ptr n-strings tok-ptr n-tokens))))
              (when (cffi:null-pointer-p sampler)
                (error 'grammar-error :grammar grammar))
              sampler))
        (dolist (ptr foreign-strings)
          (cffi:foreign-string-free ptr))))))

(defun make-infill-sampler (model)
  "Create a fill-in-the-middle sampler for FIM-capable models.
Returns a sampler pointer. Caller must free with %llama:sampler-free."
  (with-fp-traps-masked
    (let* ((vocab (%llama:model-get-vocab model))
           (sampler (%llama:sampler-init-infill vocab)))
      (when (cffi:null-pointer-p sampler)
        (error 'grammar-error :grammar "<infill>"))
      sampler)))

(defmacro with-grammar-sampler ((var model grammar &key (root "root") lazy
                                                         trigger-words
                                                         trigger-patterns
                                                         trigger-tokens)
                                &body body)
  "Create a grammar sampler, bind to VAR, execute BODY, free the sampler.
When LAZY is true, creates a lazy grammar sampler with optional trigger args."
  (let ((sampler-ptr (gensym "GRAMMAR-SAMPLER")))
    `(with-fp-traps-masked
       (let ((,sampler-ptr (if ,lazy
                               (make-grammar-sampler-lazy
                                ,model ,grammar
                                :root ,root
                                :trigger-words ,trigger-words
                                :trigger-patterns ,trigger-patterns
                                :trigger-tokens ,trigger-tokens)
                               (make-grammar-sampler ,model ,grammar :root ,root))))
         (unwind-protect
              (let ((,var ,sampler-ptr))
                ,@body)
           (%llama:sampler-free ,sampler-ptr))))))

;;; Model / context introspection wrappers

(defun read-model-buffer-string (model reader-fn &rest extra-args)
  "Read a buffer-probe string from MODEL using READER-FN.
READER-FN is called as (apply reader-fn model ...extra-args buf buf-size)."
  (let ((buf-size 256))
    (cffi:with-foreign-pointer (buf buf-size)
      (let ((n (apply reader-fn model (append extra-args (list buf buf-size)))))
        (when (< n 0)
          (return-from read-model-buffer-string nil))
        (when (>= n buf-size)
          (let ((retry-size (1+ n)))
            (cffi:with-foreign-pointer (buf2 retry-size)
              (let ((n2 (apply reader-fn model (append extra-args (list buf2 retry-size)))))
                (return-from read-model-buffer-string
                  (cffi:foreign-string-to-lisp buf2 :count (max 0 n2)))))))
        (cffi:foreign-string-to-lisp buf :count n)))))

(defun model-description (model)
  "Return MODEL's description as a string."
  (with-fp-traps-masked
    (read-model-buffer-string model #'%llama:model-desc)))

(defun model-metadata (model)
  "Return all metadata from MODEL as an alist of (key . value) string pairs."
  (with-fp-traps-masked
    (let ((count (%llama:model-meta-count model)))
      (loop for i from 0 below count
            collect (cons
                     (read-model-buffer-string
                      model #'%llama:model-meta-key-by-index i)
                     (read-model-buffer-string
                      model #'%llama:model-meta-val-str-by-index i))))))

(defun model-info (model)
  "Return a plist of MODEL's numeric and boolean properties."
  (with-fp-traps-masked
    (list :n-params (%llama:model-n-params model)
          :n-layers (%llama:model-n-layer model)
          :n-ctx-train (%llama:model-n-ctx-train model)
          :size-bytes (%llama:model-size model)
          :n-heads (%llama:model-n-head model)
          :n-heads-kv (%llama:model-n-head-kv model)
          :n-embd-in (%llama:model-n-embd-inp model)
          :n-embd-out (%llama:model-n-embd-out model)
          :n-swa (%llama:model-n-swa model)
          :rope-type (%llama:model-rope-type model)
          :rope-freq-scale (%llama:model-rope-freq-scale-train model)
          :n-cls-out (%llama:model-n-cls-out model)
          :encoder-p (not (zerop (%llama:model-has-encoder model)))
          :decoder-p (not (zerop (%llama:model-has-decoder model)))
          :recurrent-p (not (zerop (%llama:model-is-recurrent model)))
          :hybrid-p (not (zerop (%llama:model-is-hybrid model)))
          :diffusion-p (not (zerop (%llama:model-is-diffusion model))))))

(defun model-cls-label (model index)
  "Return the classification label string at INDEX, or NIL if unavailable."
  (with-fp-traps-masked
    (%llama:model-cls-label model index)))

(defun context-info (ctx)
  "Return a plist of CTX's configuration properties."
  (with-fp-traps-masked
    (list :n-ctx (%llama:n-ctx ctx)
          :n-batch (%llama:n-batch ctx)
          :n-ubatch (%llama:n-ubatch ctx)
          :n-seq-max (%llama:n-seq-max ctx)
          :n-threads (%llama:n-threads ctx)
          :n-threads-batch (%llama:n-threads-batch ctx)
          :pooling-type (%llama:pooling-type ctx))))

(defun system-info ()
  "Return a string describing the llama.cpp build and system capabilities."
  (with-fp-traps-masked
    (%llama:print-system-info)))

;;; Context runtime configuration

(defun set-n-threads (ctx n-threads n-threads-batch)
  "Set the number of threads on CTX for single-token decoding (N-THREADS)
and batch decoding (N-THREADS-BATCH). Both values are required together."
  (check-type n-threads (integer 0 *))
  (check-type n-threads-batch (integer 0 *))
  (with-fp-traps-masked
    (%llama:set-n-threads ctx n-threads n-threads-batch))
  nil)

(defun set-warmup (ctx warmup-p)
  "Enable or disable the warmup pass on CTX."
  (with-fp-traps-masked
    (%llama:set-warmup ctx (%bool->c warmup-p)))
  nil)

(defun set-causal-attn (ctx causal-attn-p)
  "Enable or disable causal attention on CTX."
  (with-fp-traps-masked
    (%llama:set-causal-attn ctx (%bool->c causal-attn-p)))
  nil)

(defun set-embeddings (ctx embeddings-p)
  "Set the embedding-output flag on CTX to EMBEDDINGS-P. Must match the mode
the context was created with: enable only on contexts created with :embeddings
non-nil, disable only on contexts created without it. Toggling on a mismatched
context leaves internal C state inconsistent."
  (with-fp-traps-masked
    (%llama:set-embeddings ctx (%bool->c embeddings-p)))
  nil)

(defun synchronize (ctx)
  "Block until all pending async operations on CTX have completed."
  (with-fp-traps-masked
    (%llama:synchronize ctx))
  nil)

(defun set-abort-callback (ctx callback &optional data)
  "Register an abort callback on CTX. CALLBACK must be a foreign function
pointer obtained via CFFI:CALLBACK, or NIL to clear the callback.
DATA is an optional opaque data pointer passed to the callback."
  (with-fp-traps-masked
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
  (with-fp-traps-masked
    (%llama:attach-threadpool
     ctx
     threadpool
     (or threadpool-batch (cffi:null-pointer))))
  nil)

(defun detach-threadpool (ctx)
  "Detach the threadpool from CTX. Does not free the threadpool."
  (with-fp-traps-masked
    (%llama:detach-threadpool ctx))
  nil)

;;; Performance counters

(defun context-perf (ctx)
  "Return performance data for CTX as a plist.
Keys: :T-START-MS :T-LOAD-MS :T-P-EVAL-MS :T-EVAL-MS :N-P-EVAL :N-EVAL :N-REUSED"
  (with-fp-traps-masked
    (let ((data (%llama:perf-context ctx)))
      (list :t-start-ms  (getf data '%llama::t-start-ms)
            :t-load-ms   (getf data '%llama::t-load-ms)
            :t-p-eval-ms (getf data '%llama::t-p-eval-ms)
            :t-eval-ms   (getf data '%llama::t-eval-ms)
            :n-p-eval    (getf data '%llama::n-p-eval)
            :n-eval      (getf data '%llama::n-eval)
            :n-reused    (getf data '%llama::n-reused)))))

(defun print-context-perf (ctx)
  "Print context performance statistics for CTX to stderr."
  (with-fp-traps-masked
    (%llama:perf-context-print ctx))
  nil)

(defun reset-context-perf (ctx)
  "Reset context performance counters for CTX."
  (with-fp-traps-masked
    (%llama:perf-context-reset ctx))
  nil)

(defun sampler-perf (chain)
  "Return performance data for sampler CHAIN as a plist.
Keys: :T-SAMPLE-MS :N-SAMPLE"
  (with-fp-traps-masked
    (let ((data (%llama:perf-sampler chain)))
      (list :t-sample-ms (getf data '%llama::t-sample-ms)
            :n-sample    (getf data '%llama::n-sample)))))

(defun print-sampler-perf (chain)
  "Print sampler performance statistics for CHAIN to stderr."
  (with-fp-traps-masked
    (%llama:perf-sampler-print chain))
  nil)

(defun reset-sampler-perf (chain)
  "Reset sampler performance counters for CHAIN."
  (with-fp-traps-masked
    (%llama:perf-sampler-reset chain))
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

(cffi:defcallback %log-dispatcher :void
    ((level :int) (text :string) (data :pointer))
  (declare (ignore data))
  (when *log-callback*
    (ignore-errors (funcall *log-callback* level text))))

(defun set-log-callback (fn)
  "Set FN as the Lisp log callback for all llama.cpp log messages.
FN is called as (fn level text) where LEVEL is an integer (1=debug
2=info 3=warn 4=error) and TEXT is the message string.
Pass NIL to restore the default C stderr logger."
  (setf *log-callback* fn)
  (with-fp-traps-masked
    (%llama:log-set
     (if fn (cffi:callback %log-dispatcher) (cffi:null-pointer))
     (cffi:null-pointer)))
  nil)

(defun get-log-callback ()
  "Return the current Lisp log callback, or NIL if unset."
  *log-callback*)

;;; System queries

(defun time-us ()
  "Return the current wall-clock time in microseconds."
  (with-fp-traps-masked
    (%llama:time-us)))

(defun system-capabilities ()
  "Return a plist of system capability flags.
Keys: :MMAP :MLOCK :GPU-OFFLOAD :RPC :MAX-DEVICES"
  (with-fp-traps-masked
    (list :mmap        (not (zerop (%llama:supports-mmap)))
          :mlock       (not (zerop (%llama:supports-mlock)))
          :gpu-offload (not (zerop (%llama:supports-gpu-offload)))
          :rpc         (not (zerop (%llama:supports-rpc)))
          :max-devices (%llama:max-devices))))

;;; Sampler utilities

(defun sampler-seed (sampler)
  "Return the current RNG seed from SAMPLER as an integer."
  (with-fp-traps-masked
    (%llama:sampler-get-seed sampler)))

;;; Batch API wrappers

(defstruct (%batch-handle (:constructor %make-batch-handle)
                           (:conc-name %batch-))
  (capacity 0 :type fixnum)
  (n-embd 0 :type fixnum)
  (data nil :type list))

(defun %batch-check-overflow (batch)
  (let ((count (getf (%batch-data batch) '%llama:n-tokens))
        (cap (%batch-capacity batch)))
    (when (>= count cap)
      (error 'batch-overflow-error :capacity cap :token-count count))))

(defmacro with-batch ((var n-tokens &key (n-embd 0) (n-seq-max 1)) &body body)
  "Allocate a batch with capacity for N-TOKENS, bind to VAR, execute BODY, free.
N-EMBD when non-zero allocates embedding slots instead of token slots.
N-SEQ-MAX is the maximum number of sequences per token (default 1).
Signals BATCH-INIT-ERROR if N-TOKENS <= 0 or allocation fails."
  (let ((cap (gensym "CAP"))
        (embd-val (gensym "EMBD"))
        (seq-max-val (gensym "SEQ-MAX"))
        (plist (gensym "PLIST"))
        (handle (gensym "HANDLE")))
    `(with-fp-traps-masked
       (let* ((,cap ,n-tokens)
              (,embd-val ,n-embd)
              (,seq-max-val ,n-seq-max))
         (when (<= ,cap 0)
           (error 'batch-init-error :n-tokens ,cap))
         (let ((,plist (%llama:batch-init ,cap ,embd-val ,seq-max-val)))
           (let ((key-ptr (if (zerop ,embd-val)
                              (getf ,plist '%llama:token)
                              (getf ,plist '%llama:embd))))
             (when (cffi:null-pointer-p key-ptr)
               (error 'batch-init-error :n-tokens ,cap)))
           (let ((,handle (%make-batch-handle :capacity ,cap
                                              :n-embd ,embd-val
                                              :data ,plist)))
             (unwind-protect
                  (let ((,var ,handle))
                    ,@body)
               (%llama:batch-free (%batch-data ,handle)))))))))

(defun batch-token-count (batch)
  "Return the current number of tokens in BATCH."
  (getf (%batch-data batch) '%llama:n-tokens))

(defun batch-clear (batch)
  "Reset BATCH's active token count to 0.
Existing slot contents remain allocated and will be overwritten
by subsequent batch-add-token or batch-add-embedding calls."
  (setf (getf (%batch-data batch) '%llama:n-tokens) 0)
  nil)

(defun batch-add-token (batch token pos seq-ids &key logits)
  "Add one token to BATCH at position POS for the given sequence(s).
TOKEN is a token integer. POS is the position integer.
SEQ-IDS is an integer (single sequence) or a list of integers (multi-sequence).
LOGITS when true requests logit computation for this token.
Signals BATCH-OVERFLOW-ERROR if BATCH is at capacity."
  (%batch-check-overflow batch)
  (let* ((plist (%batch-data batch))
         (idx (getf plist '%llama:n-tokens))
         (tok-ptr (getf plist '%llama:token))
         (pos-ptr (getf plist '%llama:pos))
         (n-seq-ptr (getf plist '%llama:n-seq-id))
         (seq-ptr (getf plist '%llama:seq-id))
         (logits-ptr (getf plist '%llama:logits))
         (seq-list (if (listp seq-ids) seq-ids (list seq-ids)))
         (n-seq (length seq-list)))
    (setf (cffi:mem-aref tok-ptr '%llama:token idx) token)
    (setf (cffi:mem-aref pos-ptr '%llama:pos idx) pos)
    (setf (cffi:mem-aref n-seq-ptr '%llama:int32-t idx) n-seq)
    (let ((seq-arr (cffi:mem-aref seq-ptr :pointer idx)))
      (loop for s in seq-list
            for j from 0
            do (setf (cffi:mem-aref seq-arr '%llama:seq-id j) s)))
    (setf (cffi:mem-aref logits-ptr '%llama:int8-t idx) (if logits 1 0))
    (setf (getf (%batch-data batch) '%llama:n-tokens) (1+ idx)))
  nil)

(defun batch-add-embedding (batch embedding pos seq-ids &key logits)
  "Add one embedding vector to BATCH at position POS for the given sequence(s).
EMBEDDING is a sequence of floats whose length must match the N-EMBD
used when creating the batch.
SEQ-IDS is an integer (single sequence) or a list of integers (multi-sequence).
LOGITS when true requests logit computation for this slot.
Signals BATCH-OVERFLOW-ERROR if BATCH is at capacity."
  (%batch-check-overflow batch)
  (let* ((plist (%batch-data batch))
         (n-embd (%batch-n-embd batch))
         (idx (getf plist '%llama:n-tokens))
         (embd-ptr (getf plist '%llama:embd))
         (pos-ptr (getf plist '%llama:pos))
         (n-seq-ptr (getf plist '%llama:n-seq-id))
         (seq-ptr (getf plist '%llama:seq-id))
         (logits-ptr (getf plist '%llama:logits))
         (seq-list (if (listp seq-ids) seq-ids (list seq-ids)))
         (n-seq (length seq-list))
         (embd-offset (* idx n-embd)))
    (when (/= (length embedding) n-embd)
      (error "Embedding length ~D does not match batch n-embd ~D"
             (length embedding) n-embd))
    (etypecase embedding
      (vector (dotimes (j n-embd)
                (setf (cffi:mem-aref embd-ptr :float (+ embd-offset j))
                      (coerce (aref embedding j) 'single-float))))
      (list (loop for v in embedding
                  for j from 0
                  do (setf (cffi:mem-aref embd-ptr :float (+ embd-offset j))
                           (coerce v 'single-float)))))
    (setf (cffi:mem-aref pos-ptr '%llama:pos idx) pos)
    (setf (cffi:mem-aref n-seq-ptr '%llama:int32-t idx) n-seq)
    (let ((seq-arr (cffi:mem-aref seq-ptr :pointer idx)))
      (loop for s in seq-list
            for j from 0
            do (setf (cffi:mem-aref seq-arr '%llama:seq-id j) s)))
    (setf (cffi:mem-aref logits-ptr '%llama:int8-t idx) (if logits 1 0))
    (setf (getf (%batch-data batch) '%llama:n-tokens) (1+ idx)))
  nil)

(defun batch-add-sequence (batch tokens seq-id &key (start-pos 0) (logits :last))
  "Add a sequence of tokens to BATCH with sequential positions.
TOKENS is a vector of token integers. SEQ-ID is the sequence identifier.
START-POS is the first position (default 0).
LOGITS controls which tokens get logit computation:
  :LAST (default) — only the final token
  :ALL — every token
  NIL — no tokens
Signals BATCH-OVERFLOW-ERROR if BATCH would exceed capacity."
  (let ((n (length tokens)))
    (dotimes (i n)
      (batch-add-token batch (aref tokens i) (+ start-pos i) seq-id
                       :logits (ecase logits
                                 (:last (= i (1- n)))
                                 (:all t)
                                 ((nil) nil)))))
  nil)

(defun batch-decode (ctx batch)
  "Decode BATCH using context CTX.
Signals DECODE-ERROR on failure. Returns NIL on success."
  (with-fp-traps-masked
    (let ((rc (%llama:decode ctx (%batch-data batch))))
      (unless (zerop rc)
        (error 'decode-error :code rc))
      nil)))

(defun batch-encode (ctx batch)
  "Encode BATCH using context CTX.
Signals DECODE-ERROR on failure. Returns NIL on success."
  (with-fp-traps-masked
    (let ((rc (%llama:encode ctx (%batch-data batch))))
      (unless (zerop rc)
        (error 'decode-error :code rc))
      nil)))

(defun generate-parallel (ctx prompts &key (max-tokens 256) (temp 0.8)
                                           top-k top-p min-p (seed 42)
                                           (parse-special t)
                                           typical-p
                                           xtc-probability xtc-threshold
                                           top-n-sigma
                                           mirostat mirostat-v2
                                           (mirostat-tau 5.0) (mirostat-eta 0.1)
                                           repeat-penalty frequency-penalty
                                           presence-penalty (penalty-last-n 64)
                                           dry-multiplier (dry-base 1.75)
                                           (dry-allowed-length 2)
                                           (dry-penalty-last-n -1)
                                           dry-seq-breakers
                                           logit-bias
                                           dynamic-temp-range
                                           (dynamic-temp-exponent 1.0)
                                           adaptive-p (adaptive-p-decay 0.0))
  "Generate text for multiple PROMPTS in parallel using the batch API.
Each element of PROMPTS is a string or a pre-tokenized token vector.
All sequences share the same sampler configuration; each gets a unique
seed derived from SEED + sequence-index for independent sampling.
The context must be created with :N-SEQ-MAX >= (length prompts).
Returns two values: a list of generated strings and a list of stop
reasons (:eog or :length) corresponding to each prompt."
  (when (endp prompts)
    (return-from generate-parallel (values nil nil)))
  (with-fp-traps-masked
    (let* ((model (%llama:get-model ctx))
           (vocab (%llama:model-get-vocab model))
           (n-seq (length prompts))
           (token-vecs (mapcar (lambda (p)
                                 (etypecase p
                                   (string (tokenize model p
                                                     :parse-special parse-special))
                                   (vector p)))
                               prompts))
           (total-prompt-tokens (reduce #'+ token-vecs :key #'length))
           (gen-tokens (loop repeat n-seq
                             collect (make-array 0 :element-type 'fixnum
                                                   :adjustable t
                                                   :fill-pointer 0)))
           (positions (mapcar #'length token-vecs))
           (active (make-list n-seq :initial-element t))
           (samplers '()))
      (%llama:memory-clear (%llama:get-memory ctx) 1)
      (unwind-protect
           (progn
             (dotimes (seq n-seq)
               (push (build-sampler-chain
                      :temp temp :top-k top-k :top-p top-p
                      :min-p min-p :seed (+ seed seq)
                      :model model
                      :typical-p typical-p
                      :xtc-probability xtc-probability
                      :xtc-threshold xtc-threshold
                      :top-n-sigma top-n-sigma
                      :mirostat mirostat :mirostat-v2 mirostat-v2
                      :mirostat-tau mirostat-tau :mirostat-eta mirostat-eta
                      :repeat-penalty repeat-penalty
                      :frequency-penalty frequency-penalty
                      :presence-penalty presence-penalty
                      :penalty-last-n penalty-last-n
                      :dry-multiplier dry-multiplier :dry-base dry-base
                      :dry-allowed-length dry-allowed-length
                      :dry-penalty-last-n dry-penalty-last-n
                      :dry-seq-breakers dry-seq-breakers
                      :logit-bias logit-bias
                      :dynamic-temp-range dynamic-temp-range
                      :dynamic-temp-exponent dynamic-temp-exponent
                      :adaptive-p adaptive-p
                      :adaptive-p-decay adaptive-p-decay)
                     samplers))
             (setf samplers (nreverse samplers))
             (with-batch (batch total-prompt-tokens)
               (loop for tokens in token-vecs
                     for seq from 0
                     do (batch-add-sequence batch tokens seq :logits :last))
               (batch-decode ctx batch)
               (let ((logit-indices
                       (let ((acc 0))
                         (mapcar (lambda (tv)
                                   (prog1 (+ acc (1- (length tv)))
                                     (incf acc (length tv))))
                                 token-vecs))))
                 (dotimes (step max-tokens)
                   (unless (some #'identity active) (return))
                   (let ((new-tokens
                           (loop for seq from 0 below n-seq
                                 for smpl in samplers
                                 for idx in logit-indices
                                 collect (when (nth seq active)
                                           (%llama:sampler-sample
                                            smpl ctx idx)))))
                     (loop for tok in new-tokens
                           for seq from 0
                           when (and tok (nth seq active))
                           do (if (not (zerop (%llama:token-is-eog
                                              vocab tok)))
                                  (setf (nth seq active) nil)
                                  (vector-push-extend
                                   tok (nth seq gen-tokens))))
                     (when (some #'identity active)
                       (batch-clear batch)
                       (let ((batch-idx 0))
                         (loop for tok in new-tokens
                               for seq from 0
                               when (and tok (nth seq active))
                               do (batch-add-token batch tok
                                                   (nth seq positions)
                                                   seq :logits t)
                                  (setf (nth seq logit-indices) batch-idx)
                                  (incf (nth seq positions))
                                  (incf batch-idx)))
                       (batch-decode ctx batch)))))))
        (dolist (s samplers)
          (%llama:sampler-free s)))
      (values (loop for tokens in gen-tokens
                    collect (if (zerop (length tokens))
                                ""
                                (detokenize model tokens
                                            :remove-special t)))
              (loop for is-active in active
                    collect (if is-active :length :eog))))))
