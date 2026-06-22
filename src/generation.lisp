(in-package #:cl-llama-cpp)

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
  (let* ((model-ptr (when model (llama-model-pointer model)))
         (chain (%llama:sampler-chain-init
                 (%llama:sampler-chain-default-params))))
    ;; 1. Logit bias (modifies logits first)
    (when logit-bias
      (let* ((vocab (%llama:model-get-vocab model-ptr))
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
      (let* ((vocab (%llama:model-get-vocab model-ptr))
             (n-ctx-train (%llama:model-n-ctx-train model-ptr))
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
       chain (llama-sampler-pointer
              (if grammar-lazy
                  (make-grammar-sampler-lazy model grammar
                    :root grammar-root
                    :trigger-words grammar-trigger-words
                    :trigger-patterns grammar-trigger-patterns
                    :trigger-tokens grammar-trigger-tokens)
                  (make-grammar-sampler model grammar :root grammar-root)))))
    (when infill
      (unless model (error ":INFILL requires :MODEL"))
      (%llama:sampler-chain-add chain (llama-sampler-pointer (make-infill-sampler model))))
    ;; 5. Sampling strategy
    (cond
      (greedy
       (%llama:sampler-chain-add chain (%llama:sampler-init-greedy)))
      ((or mirostat mirostat-v2)
       (if mirostat
           (let* ((vocab (%llama:model-get-vocab model-ptr))
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
  "Allocate a sampler chain, bind it to VAR, execute BODY, then free it.

With no keyword arguments, an empty chain is created — use SAMPLER-CHAIN-ADD
inside BODY to populate it.  With keyword arguments the chain is pre-built by
BUILD-SAMPLER-CHAIN (same keywords as GENERATE)."
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
  (let ((chain-sym (gensym "CHAIN")))
    (if args
        `(with-llama-compatible-fp-environment
           (let ((,chain-sym (build-sampler-chain ,@args)))
             (unwind-protect
                  (let ((,var (%make-llama-sampler :pointer ,chain-sym)))
                    ,@body)
               (%llama:sampler-free ,chain-sym))))
        `(with-llama-compatible-fp-environment
           (let ((,chain-sym (%llama:sampler-chain-init
                              (%llama:sampler-chain-default-params))))
             (unwind-protect
                  (let ((,var (%make-llama-sampler :pointer ,chain-sym)))
                    ,@body)
               (%llama:sampler-free ,chain-sym)))))))

(defun sampler-chain-add (chain sampler)
  "Add SAMPLER to CHAIN.  Both arguments may be typed LLAMA-SAMPLER handles or
raw CFFI pointers (e.g. the return value of %LLAMA:SAMPLER-INIT-TEMP).  The
chain takes ownership of the sampler — freeing the chain frees all added
samplers."
  (let ((chain-ptr (if (llama-sampler-p chain) (llama-sampler-pointer chain) chain))
        (smpl-ptr  (if (llama-sampler-p sampler) (llama-sampler-pointer sampler) sampler)))
    (with-llama-compatible-fp-environment
      (%llama:sampler-chain-add chain-ptr smpl-ptr))))

;;; Sampler utilities

(defun sampler-seed (sampler)
  "Return the current RNG seed from SAMPLER as an integer."
  (with-llama-compatible-fp-environment
    (%llama:sampler-get-seed (llama-sampler-pointer sampler))))

;;; Individual sampler constructors
;;; Each returns a LLAMA-SAMPLER handle. The caller owns it and must either
;;; free it with FREE-SAMPLER or add it to a chain (which then owns it).

(defun make-greedy-sampler ()
  "Create a greedy sampler that always picks the highest-probability token."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-greedy))))

(defun make-dist-sampler (&optional (seed 42))
  "Create a distribution sampler (random sampling weighted by probabilities).
SEED is a uint32 random seed."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-dist seed))))

(defun make-top-k-sampler (k)
  "Create a top-k sampler restricting candidates to the K highest-probability tokens."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-top-k k))))

(defun make-top-p-sampler (p &optional (min-keep 1))
  "Create a top-p (nucleus) sampler keeping tokens until cumulative probability >= P."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-top-p (coerce p 'single-float) min-keep))))

(defun make-min-p-sampler (p &optional (min-keep 1))
  "Create a min-p sampler removing tokens with probability < P * max-prob."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-min-p (coerce p 'single-float) min-keep))))

(defun make-typical-sampler (p &optional (min-keep 1))
  "Create a locally typical sampler with probability mass P."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-typical (coerce p 'single-float) min-keep))))

(defun make-temp-sampler (temp)
  "Create a temperature sampler scaling logits by TEMP before softmax."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-temp (coerce temp 'single-float)))))

(defun make-temp-ext-sampler (temp delta &optional (exponent 1.0))
  "Create an extended temperature sampler with dynamic temperature range.
TEMP is the base temperature, DELTA the range, EXPONENT the curve shape."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-temp-ext
                                   (coerce temp 'single-float)
                                   (coerce delta 'single-float)
                                   (coerce exponent 'single-float)))))

(defun make-xtc-sampler (probability threshold &optional (min-keep 1) (seed 42))
  "Create an XTC sampler that trims high-probability tokens exceeding THRESHOLD.
PROBABILITY is the chance of applying XTC per sampling step."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-xtc
                                   (coerce probability 'single-float)
                                   (coerce threshold 'single-float)
                                   min-keep seed))))

(defun make-top-n-sigma-sampler (sigma)
  "Create a top-n-sigma sampler keeping tokens within SIGMA standard deviations of the max logit."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-top-n-sigma (coerce sigma 'single-float)))))

(defun make-mirostat-v2-sampler (seed tau eta)
  "Create a Mirostat v2 sampler targeting perplexity TAU with learning rate ETA."
  (with-llama-compatible-fp-environment
    (%make-llama-sampler :pointer (%llama:sampler-init-mirostat-v2
                                   seed
                                   (coerce tau 'single-float)
                                   (coerce eta 'single-float)))))

(defun free-sampler (sampler)
  "Free a LLAMA-SAMPLER handle created by any MAKE-*-SAMPLER function.
Do not call on samplers that have been added to a chain — the chain owns those."
  (with-llama-compatible-fp-environment
    (%llama:sampler-free (llama-sampler-pointer sampler)))
  nil)

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
                                  adaptive-p (adaptive-p-decay 0.0)
                                  sampler)
  "Generate text by continuing PROMPT. Returns two values: the generated string
and a stop reason (:eog, :length, or :callback).
Uses the context's model for tokenization. Blocks until EOS or MAX-TOKENS.
Supports extended sampler keywords: :TYPICAL-P, :XTC-PROBABILITY, :XTC-THRESHOLD,
:MIROSTAT, :MIROSTAT-V2, :REPEAT-PENALTY, :FREQUENCY-PENALTY, :PRESENCE-PENALTY,
:DRY-MULTIPLIER, :LOGIT-BIAS, :TOP-N-SIGMA, :DYNAMIC-TEMP-RANGE, :ADAPTIVE-P, etc.

When :SAMPLER is provided (a LLAMA-SAMPLER handle, typically from WITH-SAMPLER-CHAIN),
GENERATE borrows the chain and does not free it — the caller owns the lifetime.
All other sampler-related keywords are ignored when :SAMPLER is supplied.

Signals INPUT-VALIDATION-ERROR if MAX-TOKENS is not a positive integer or
PROMPT is neither a string nor a vector."
  (declare (optimize (speed 3)))
  (check-type max-tokens (integer 1 *))
  (unless prompt-tokens
    (check-type prompt (or string vector)))
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (raw-model (%llama:get-model ctx-ptr))
           (model (%make-llama-model :pointer raw-model))
           (vocab (%llama:model-get-vocab raw-model))
           (prompt-tokens (or prompt-tokens
                              (tokenize model prompt :parse-special parse-special)))
           (n-prompt (length prompt-tokens))
           (generated (make-array 0 :element-type 'fixnum
                                    :adjustable t :fill-pointer 0)))
      (declare (type fixnum n-prompt)
               (type (vector fixnum) generated))
      (%llama:memory-clear (%llama:get-memory ctx-ptr) 1)
      ;; Decode the prompt
      (cffi:with-foreign-object (tok-buf '%llama:token n-prompt)
        (dotimes (i n-prompt)
          (declare (type fixnum i))
          (setf (cffi:mem-aref tok-buf '%llama:token i) (aref prompt-tokens i)))
        (let* ((batch (%llama:batch-get-one tok-buf n-prompt))
               (rc (%llama:decode ctx-ptr batch)))
          (unless (zerop rc)
            (error 'decode-error :code rc)))
        (setf (llama-context-compute-pending-p ctx) t))
      ;; Warn if caller supplied a chain but also passed sampler-building kwargs
      (when (and sampler
                 (or grammar top-k top-p min-p typical-p xtc-probability
                     dry-multiplier logit-bias mirostat mirostat-v2
                     repeat-penalty frequency-penalty presence-penalty
                     adaptive-p dynamic-temp-range top-n-sigma))
        (warn "~@<When :SAMPLER is provided, other sampler keywords ~
(:GRAMMAR, :TOP-K, :TEMP, etc.) are ignored.~@:>"))
      ;; Generation loop
      (let ((chain-ptr (if sampler
                           (llama-sampler-pointer sampler)
                           (build-sampler-chain
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
                            :adaptive-p-decay adaptive-p-decay)))
            (emitted-len 0)
            (stop-reason nil))
        (declare (type fixnum emitted-len))
        (unwind-protect
            ;; sampler-sample already calls sampler-accept internally — do NOT
            ;; call sampler-accept again or the grammar FSM double-advances.
            (loop for i of-type fixnum from 0 below max-tokens
                  for new-token of-type fixnum = (%llama:sampler-sample chain-ptr ctx-ptr -1)
                  until (not (zerop (%llama:token-is-eog vocab new-token)))
                  do (vector-push-extend new-token generated)
                     (when token-callback
                       ;; detokenize is a library concern — errors propagate naturally
                       (let* ((full (detokenize model generated :remove-special t))
                              (new-text (subseq full emitted-len)))
                         (when (plusp (length new-text))
                           (setf emitted-len (length full))
                           ;; callback is the user's concern — separate handler boundary
                           (restart-case
                               (handler-bind
                                   ((error (lambda (c)
                                             (declare (ignore c))
                                             (invoke-restart 'abort-generation))))
                                 (unless (funcall token-callback new-text)
                                   (setf stop-reason :callback)
                                   (loop-finish)))
                             (ignore-callback-error ()
                               :report "Ignore the callback error and continue generation"
                               nil)
                             (abort-generation ()
                               :report "Abort generation due to token-callback error"
                               (setf stop-reason :error)
                               (loop-finish))))))
                     (cffi:with-foreign-object (tok-buf '%llama:token 1)
                       (setf (cffi:mem-aref tok-buf '%llama:token 0) new-token)
                       (let* ((batch (%llama:batch-get-one tok-buf 1))
                              (rc (%llama:decode ctx-ptr batch)))
                         (declare (type fixnum rc))
                         (unless (zerop rc)
                           (error 'decode-error :code rc)))
                       (setf (llama-context-compute-pending-p ctx) t)))
          (unless sampler
            (%llama:sampler-free chain-ptr)))
      ;; Convert generated tokens to string
      (let ((text (if (zerop (length generated))
                      ""
                      (let ((result-tokens (make-array (length generated)
                                                       :element-type 'fixnum)))
                        (declare (type (simple-array fixnum (*)) result-tokens))
                        (dotimes (i (length generated))
                          (declare (type fixnum i))
                          (setf (aref result-tokens i) (aref generated i)))
                        (detokenize model result-tokens :remove-special t)))))
        (values text
                (or stop-reason
                    (if (= (length generated) max-tokens) :length :eog))))))))

(defun embed (ctx text &key (normalize t))
  "Compute embeddings for TEXT. Returns a vector of single-floats.
The context must have been created with :embeddings 1.
When NORMALIZE is true (default), L2-normalizes the result.
Signals INPUT-VALIDATION-ERROR if TEXT is not a non-empty string."
  (declare (optimize (speed 3)))
  (check-type text string)
  (when (zerop (length text))
    (error 'input-validation-error
           :function-name 'embed :argument :text :value text
           :reason "text must be non-empty"))
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (raw-model (%llama:get-model ctx-ptr))
           (model (%make-llama-model :pointer raw-model))
           (tokens (tokenize model text))
           (n-tokens (length tokens))
           (n-embd (%llama:model-n-embd raw-model)))
      (declare (type fixnum n-tokens n-embd))
      ;; Check embeddings configured before calling C encode (would crash if not)
      (when (eq (%llama:pooling-type ctx-ptr) :none)
        (error "Embeddings not available — was the context created with :EMBEDDINGS enabled?"))
      ;; Build batch and encode
      (cffi:with-foreign-object (tok-buf '%llama:token n-tokens)
        (dotimes (i n-tokens)
          (declare (type fixnum i))
          (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
        (let* ((batch (%llama:batch-get-one tok-buf n-tokens))
               (rc (%llama:encode ctx-ptr batch)))
          (unless (zerop rc)
            (error 'decode-error :code rc)))
        (setf (llama-context-compute-pending-p ctx) t))
      ;; Sync before reading — encode may run asynchronously on GPU
      (when (llama-context-compute-pending-p ctx)
        (%llama:synchronize ctx-ptr)
        (setf (llama-context-compute-pending-p ctx) nil))
      ;; Read embeddings (null-pointer → error per NIL↔null convention)
      (let* ((embd-ptr (%llama:get-embeddings-ith ctx-ptr 0)))
        (when (cffi:null-pointer-p embd-ptr)
          (error "Embeddings not available — was the context created with :EMBEDDINGS enabled?"))
        (let ((result (make-array n-embd :element-type 'single-float)))
          (declare (type (simple-array single-float (*)) result))
          (dotimes (i n-embd)
            (declare (type fixnum i))
            (setf (aref result i) (cffi:mem-aref embd-ptr :float i)))
          (when normalize
            (let ((norm (sqrt (loop for x of-type single-float across result
                                    sum (the single-float (* x x))
                                    of-type single-float))))
              (declare (type single-float norm))
              (when (> norm 0.0)
                (dotimes (i n-embd)
                  (declare (type fixnum i))
                  (setf (aref result i) (/ (aref result i) norm))))))
          result)))))
