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
    `(with-llama-compatible-fp-environment
       (let ((,chain (build-sampler-chain ,@args)))
         (unwind-protect
              (let ((,var ,chain))
                ,@body)
           (%llama:sampler-free ,chain))))))

;;; Sampler utilities

(defun sampler-seed (sampler)
  "Return the current RNG seed from SAMPLER as an integer."
  (with-llama-compatible-fp-environment
    (%llama:sampler-get-seed sampler)))

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
  (with-llama-compatible-fp-environment
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
  (with-llama-compatible-fp-environment
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
