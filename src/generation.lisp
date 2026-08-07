(in-package #:cl-llama-cpp)

;;; Token history ring buffer

(defstruct (token-history (:constructor %make-token-history))
  (buffer (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (capacity 0 :type fixnum)
  (head 0 :type fixnum)
  (count 0 :type fixnum))

(defun make-token-history (capacity)
  "Create a token history ring buffer with the given CAPACITY."
  (declare (optimize (speed 3)))
  (check-type capacity (integer 1 *))
  (%make-token-history
   :buffer (make-array capacity :element-type 'fixnum :initial-element 0)
   :capacity capacity))

(defun token-history-push (history token)
  "Push TOKEN into HISTORY, overwriting the oldest entry if full."
  (declare (optimize (speed 3))
           (type fixnum token))
  (let ((cap (token-history-capacity history))
        (head (token-history-head history)))
    (declare (type fixnum cap head))
    (setf (aref (token-history-buffer history) head) token)
    (setf (token-history-head history) (mod (1+ head) cap))
    (when (< (token-history-count history) cap)
      (incf (token-history-count history)))
    token))

(defun token-history-last (history)
  "Return the most recently pushed token, or NIL if history is empty."
  (declare (optimize (speed 3)))
  (let ((count (token-history-count history)))
    (declare (type fixnum count))
    (when (plusp count)
      (let ((idx (mod (1- (token-history-head history))
                      (token-history-capacity history))))
        (declare (type fixnum idx))
        (aref (token-history-buffer history) idx)))))

(defun token-history-tokens (history n)
  "Return the last min(N, available) tokens as a (simple-array fixnum (*)),
oldest first."
  (declare (optimize (speed 3))
           (type fixnum n))
  (let* ((count (token-history-count history))
         (actual (min n count))
         (cap (token-history-capacity history))
         (buf (token-history-buffer history))
         (head (token-history-head history))
         (result (make-array actual :element-type 'fixnum)))
    (declare (type fixnum count actual cap head)
             (type (simple-array fixnum (*)) buf result))
    (let ((start (mod (- head actual) cap)))
      (declare (type fixnum start))
      (dotimes (i actual)
        (declare (type fixnum i))
        (setf (aref result i) (aref buf (mod (+ start i) cap)))))
    result))

(defun token-history-clear (history)
  "Clear all tokens from HISTORY."
  (setf (token-history-head history) 0
        (token-history-count history) 0)
  nil)

(defun token-history-prev-str (history model n)
  "Return the detokenized string of the last N tokens in HISTORY."
  (let ((tokens (token-history-tokens history n)))
    (if (zerop (length tokens))
        ""
        (detokenize model tokens :remove-special t))))

;;; Seed resolution

(defun resolve-seed (seed)
  "Resolve a seed argument to a uint32 integer for the C sampler layer.
INTEGER → passed through unchanged.
:RANDOM or NIL → %LLAMA:+DEFAULT-SEED+ (0xFFFFFFFF), telling the C layer
to draw a nondeterministic seed internally.
Any other type signals INPUT-VALIDATION-ERROR."
  (declare (optimize (speed 3)))
  (cond
    ((typep seed '(integer 0 4294967295)) seed)
    ((or (eq seed :random) (null seed)) %llama:+default-seed+)
    (t (error 'input-validation-error
              :function-name 'resolve-seed :argument :seed :value seed
              :reason "seed must be an integer (0–4294967295), :RANDOM, or NIL"))))

(defun make-sampler-config (&rest kwargs
                            &key temp top-k top-p min-p seed greedy
                                 grammar grammar-root grammar-lazy
                                 grammar-trigger-words grammar-trigger-patterns
                                 grammar-trigger-tokens infill
                                 grammar-first
                                 typical-p xtc-probability xtc-threshold
                                 top-n-sigma mirostat mirostat-v2
                                 mirostat-tau mirostat-eta
                                 repeat-penalty frequency-penalty
                                 presence-penalty penalty-last-n
                                 dry-multiplier dry-base dry-allowed-length
                                 dry-penalty-last-n dry-seq-breakers
                                 logit-bias dynamic-temp-range dynamic-temp-exponent
                                 adaptive-p adaptive-p-decay)
  "Bundle sampler parameters into a reusable config plist.

The returned plist can be passed as :SAMPLER-CONFIG to GENERATE,
BUILD-SAMPLER-CHAIN, or WITH-SAMPLER-CHAIN.  The config provides
defaults; any keyword supplied at the call site overrides the
corresponding config entry.  Only the parameters you explicitly supply
here are stored — the rest continue to use each function's own
built-in defaults.

SEED may be an integer, :RANDOM, or NIL.  Sentinels are stored verbatim;
resolution to a concrete seed happens when the config is consumed."
  (declare (ignore temp top-k top-p min-p seed greedy
                   grammar grammar-root grammar-lazy
                   grammar-trigger-words grammar-trigger-patterns
                   grammar-trigger-tokens infill
                   grammar-first
                   typical-p xtc-probability xtc-threshold
                   top-n-sigma mirostat mirostat-v2
                   mirostat-tau mirostat-eta
                   repeat-penalty frequency-penalty
                   presence-penalty penalty-last-n
                   dry-multiplier dry-base dry-allowed-length
                   dry-penalty-last-n dry-seq-breakers
                   logit-bias dynamic-temp-range dynamic-temp-exponent
                   adaptive-p adaptive-p-decay))
  kwargs)

(defun build-sampler-chain (&rest all-kwargs
                           &key sampler-config
                                (temp 0.8) top-k top-p min-p (seed 42) greedy
                                model grammar (grammar-root "root") grammar-lazy
                                grammar-trigger-words grammar-trigger-patterns
                                grammar-trigger-tokens infill
                                grammar-first
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
When GRAMMAR-FIRST is true and GRAMMAR is provided, the grammar sampler is
NOT added to the chain — instead it is returned as a second value so the caller
can apply it as a logit filter before or after the chain (matching the
grammar-first resampling pattern from common_sampler).
When INFILL is true, an infill sampler is added (requires MODEL).
When MIROSTAT or MIROSTAT-V2 is true, the normal top-k/top-p/min-p/temp/dist
chain is replaced with the mirostat sampler.
When SAMPLER-CONFIG is a plist (from MAKE-SAMPLER-CONFIG), it provides default
values for any sampler parameter not supplied explicitly at this call site.

SEED may be an integer (deterministic), :RANDOM (nondeterministic — the C layer
draws a fresh seed internally), or NIL (same as :RANDOM).  Default is 42."
  ;; Merge sampler-config as defaults: caller-supplied kwargs take precedence
  ;; because they appear first in the appended plist and GETF finds the first match.
  (when sampler-config
    (let* ((caller (loop for (k v) on all-kwargs by #'cddr
                         unless (eq k :sampler-config)
                         nconc (list k v)))
           (effective (append caller sampler-config)))
      (setf temp              (getf effective :temp 0.8)
            top-k             (getf effective :top-k nil)
            top-p             (getf effective :top-p nil)
            min-p             (getf effective :min-p nil)
            seed              (getf effective :seed 42)
            greedy            (getf effective :greedy nil)
            model             (getf effective :model nil)
            grammar           (getf effective :grammar nil)
            grammar-root      (getf effective :grammar-root "root")
            grammar-lazy      (getf effective :grammar-lazy nil)
            grammar-trigger-words    (getf effective :grammar-trigger-words nil)
            grammar-trigger-patterns (getf effective :grammar-trigger-patterns nil)
            grammar-trigger-tokens   (getf effective :grammar-trigger-tokens nil)
            infill            (getf effective :infill nil)
            grammar-first     (getf effective :grammar-first nil)
            typical-p         (getf effective :typical-p nil)
            xtc-probability   (getf effective :xtc-probability nil)
            xtc-threshold     (getf effective :xtc-threshold nil)
            top-n-sigma       (getf effective :top-n-sigma nil)
            mirostat          (getf effective :mirostat nil)
            mirostat-v2       (getf effective :mirostat-v2 nil)
            mirostat-tau      (getf effective :mirostat-tau 5.0)
            mirostat-eta      (getf effective :mirostat-eta 0.1)
            repeat-penalty    (getf effective :repeat-penalty nil)
            frequency-penalty (getf effective :frequency-penalty nil)
            presence-penalty  (getf effective :presence-penalty nil)
            penalty-last-n    (getf effective :penalty-last-n 64)
            dry-multiplier    (getf effective :dry-multiplier nil)
            dry-base          (getf effective :dry-base 1.75)
            dry-allowed-length    (getf effective :dry-allowed-length 2)
            dry-penalty-last-n    (getf effective :dry-penalty-last-n -1)
            dry-seq-breakers  (getf effective :dry-seq-breakers nil)
            logit-bias        (getf effective :logit-bias nil)
            dynamic-temp-range    (getf effective :dynamic-temp-range nil)
            dynamic-temp-exponent (getf effective :dynamic-temp-exponent 1.0)
            adaptive-p        (getf effective :adaptive-p nil)
            adaptive-p-decay  (getf effective :adaptive-p-decay 0.0))))
  (when (and mirostat mirostat-v2)
    (error "Cannot use both :MIROSTAT and :MIROSTAT-V2"))
  (when (and (or mirostat) (not model))
    (error ":MIROSTAT requires :MODEL"))
  (when (and dry-multiplier (not model))
    (error ":DRY-MULTIPLIER requires :MODEL"))
  (when (and logit-bias (not model))
    (error ":LOGIT-BIAS requires :MODEL"))
  (setf seed (resolve-seed seed))
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
    (let ((grammar-smpl-ptr nil))
      (when grammar
        (unless model (error ":GRAMMAR requires :MODEL"))
        (let ((gptr (llama-sampler-pointer
                     (if grammar-lazy
                         (make-grammar-sampler-lazy model grammar
                           :root grammar-root
                           :trigger-words grammar-trigger-words
                           :trigger-patterns grammar-trigger-patterns
                           :trigger-tokens grammar-trigger-tokens)
                         (make-grammar-sampler model grammar :root grammar-root)))))
          (if grammar-first
              (setf grammar-smpl-ptr gptr)
              (%llama:sampler-chain-add chain gptr))))
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
    (values chain grammar-smpl-ptr))))

(defmacro with-sampler-chain ((var &rest args
                                &key sampler-config
                                     (temp 0.8) top-k top-p min-p
                                     (seed 42) greedy
                                     model grammar (grammar-root "root")
                                     grammar-lazy grammar-trigger-words
                                     grammar-trigger-patterns grammar-trigger-tokens
                                     infill grammar-first
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
BUILD-SAMPLER-CHAIN (same keywords as GENERATE, plus :SAMPLER-CONFIG)."
  (declare (ignore sampler-config temp top-k top-p min-p seed greedy
                   model grammar grammar-root grammar-lazy
                   grammar-trigger-words grammar-trigger-patterns
                   grammar-trigger-tokens infill grammar-first
                   typical-p xtc-probability xtc-threshold
                   top-n-sigma mirostat mirostat-v2
                   mirostat-tau mirostat-eta
                   repeat-penalty frequency-penalty
                   presence-penalty penalty-last-n
                   dry-multiplier dry-base dry-allowed-length
                   dry-penalty-last-n dry-seq-breakers logit-bias
                   dynamic-temp-range dynamic-temp-exponent
                   adaptive-p adaptive-p-decay))
  (let ((chain-sym (gensym "CHAIN"))
        (grmr-sym (gensym "GRMR")))
    (if args
        `(with-llama-compatible-fp-environment
           (multiple-value-bind (,chain-sym ,grmr-sym)
               (build-sampler-chain ,@args)
             (unwind-protect
                  (let ((,var (%make-llama-sampler :pointer ,chain-sym)))
                    ,@body)
               (when ,grmr-sym (%llama:sampler-free ,grmr-sym))
               (%llama:sampler-free ,chain-sym))))
        `(with-llama-compatible-fp-environment
           (let ((,chain-sym (%llama:sampler-chain-init
                              (%llama:sampler-chain-default-params))))
             (unwind-protect
                  (let ((,var (%make-llama-sampler :pointer ,chain-sym)))
                    ,@body)
               (%llama:sampler-free ,chain-sym)))))))

(llama-defun sampler-chain-add (chain sampler)
  "Add SAMPLER to CHAIN.  Both arguments may be typed LLAMA-SAMPLER handles or
raw CFFI pointers (e.g. the return value of %LLAMA:SAMPLER-INIT-TEMP).  The
chain takes ownership of the sampler — freeing the chain frees all added
samplers."
  (let ((chain-ptr (if (llama-sampler-p chain) (llama-sampler-pointer chain) chain))
        (smpl-ptr  (if (llama-sampler-p sampler) (llama-sampler-pointer sampler) sampler)))
    (%llama:sampler-chain-add chain-ptr smpl-ptr)))

;;; Sampler utilities

(llama-defun sampler-seed (sampler)
  "Return the current RNG seed from SAMPLER as an integer."
  (%llama:sampler-get-seed (llama-sampler-pointer sampler)))

;;; Individual sampler constructors
;;; Each returns a LLAMA-SAMPLER handle. The caller owns it and must either
;;; free it with FREE-SAMPLER or add it to a chain (which then owns it).

(llama-defun make-greedy-sampler ()
  "Create a greedy sampler that always picks the highest-probability token."
  (%make-llama-sampler :pointer (%llama:sampler-init-greedy)))

(llama-defun make-dist-sampler (&optional (seed 42))
  "Create a distribution sampler (random sampling weighted by probabilities).
SEED is a uint32 random seed, :RANDOM, or NIL (nondeterministic)."
  (%make-llama-sampler :pointer (%llama:sampler-init-dist (resolve-seed seed))))

(llama-defun make-top-k-sampler (k)
  "Create a top-k sampler restricting candidates to the K highest-probability tokens."
  (%make-llama-sampler :pointer (%llama:sampler-init-top-k k)))

(llama-defun make-top-p-sampler (p &optional (min-keep 1))
  "Create a top-p (nucleus) sampler keeping tokens until cumulative probability >= P."
  (%make-llama-sampler :pointer (%llama:sampler-init-top-p (coerce p 'single-float) min-keep)))

(llama-defun make-min-p-sampler (p &optional (min-keep 1))
  "Create a min-p sampler removing tokens with probability < P * max-prob."
  (%make-llama-sampler :pointer (%llama:sampler-init-min-p (coerce p 'single-float) min-keep)))

(llama-defun make-typical-sampler (p &optional (min-keep 1))
  "Create a locally typical sampler with probability mass P."
  (%make-llama-sampler :pointer (%llama:sampler-init-typical (coerce p 'single-float) min-keep)))

(llama-defun make-temp-sampler (temp)
  "Create a temperature sampler scaling logits by TEMP before softmax."
  (%make-llama-sampler :pointer (%llama:sampler-init-temp (coerce temp 'single-float))))

(llama-defun make-temp-ext-sampler (temp delta &optional (exponent 1.0))
  "Create an extended temperature sampler with dynamic temperature range.
TEMP is the base temperature, DELTA the range, EXPONENT the curve shape."
  (%make-llama-sampler :pointer (%llama:sampler-init-temp-ext
                                 (coerce temp 'single-float)
                                 (coerce delta 'single-float)
                                 (coerce exponent 'single-float))))

(llama-defun make-xtc-sampler (probability threshold &optional (min-keep 1) (seed 42))
  "Create an XTC sampler that trims high-probability tokens exceeding THRESHOLD.
PROBABILITY is the chance of applying XTC per sampling step.
SEED may be an integer, :RANDOM, or NIL (nondeterministic)."
  (%make-llama-sampler :pointer (%llama:sampler-init-xtc
                                 (coerce probability 'single-float)
                                 (coerce threshold 'single-float)
                                 min-keep (resolve-seed seed))))

(llama-defun make-top-n-sigma-sampler (sigma)
  "Create a top-n-sigma sampler keeping tokens within SIGMA standard deviations of the max logit."
  (%make-llama-sampler :pointer (%llama:sampler-init-top-n-sigma (coerce sigma 'single-float))))

(llama-defun make-mirostat-v2-sampler (seed tau eta)
  "Create a Mirostat v2 sampler targeting perplexity TAU with learning rate ETA.
SEED may be an integer, :RANDOM, or NIL (nondeterministic)."
  (%make-llama-sampler :pointer (%llama:sampler-init-mirostat-v2
                                 (resolve-seed seed)
                                 (coerce tau 'single-float)
                                 (coerce eta 'single-float))))

(llama-defun free-sampler (sampler)
  "Free a LLAMA-SAMPLER handle created by any MAKE-*-SAMPLER function.
Do not call on samplers that have been added to a chain — the chain owns those."
  (%llama:sampler-free (llama-sampler-pointer sampler))
  nil)

(llama-defun prefill (ctx tokens &key (seq-id 0))
  "Decode TOKENS into sequence SEQ-ID of CTX's KV cache without sampling.
Returns the number of tokens decoded as a fixnum.
Does NOT clear the KV cache first — caller controls that (consistent with
GENERATE's :RESET-CONTEXT NIL semantics).
Signals INPUT-VALIDATION-ERROR if TOKENS is not a non-empty vector."
  (declare (optimize (speed 3))
           (type fixnum seq-id))
  (unless (and (vectorp tokens) (not (stringp tokens)))
    (error 'input-validation-error
           :function-name 'prefill :argument :tokens :value tokens
           :reason "tokens must be a non-string vector of token ids"))
  (when (zerop (length tokens))
    (error 'input-validation-error
           :function-name 'prefill :argument :tokens :value tokens
           :reason "tokens must be non-empty"))
  (let ((n (length tokens))
        (ctx-ptr (llama-context-pointer ctx)))
    (declare (type fixnum n))
    (if (zerop seq-id)
        ;; Fast path for seq-id 0: batch-get-one uses null positions (auto-advances
        ;; from current KV cache end) and null seq_id (defaults to 0).
        (cffi:with-foreign-object (tok-buf '%llama:token n)
          (dotimes (i n)
            (declare (type fixnum i))
            (setf (cffi:mem-aref tok-buf '%llama:token i)
                  (aref tokens i)))
          (let* ((batch (%llama:batch-get-one tok-buf n))
                 (rc (%llama:decode ctx-ptr batch)))
            (unless (zerop rc)
              (error 'decode-error :code rc))))
        ;; Full batch path for non-zero seq-id: supply explicit start position.
        (let ((start-pos (multiple-value-bind (mn mx)
                             (kv-cache-pos ctx seq-id)
                           (if (>= mn mx) 0 (1+ mx)))))
          (with-batch (batch n)
            (batch-add-sequence batch tokens seq-id :start-pos start-pos :logits :last)
            (batch-decode ctx batch))))
    (setf (llama-context-compute-pending-p ctx) t)
    n))

(defun %copy-logits (ctx-ptr n-vocab)
  "Copy the N-VOCAB raw logits at CTX-PTR's last position (index -1) into
a fresh (simple-array single-float (*)). The FFI buffer backing these
logits is only valid until the next decode call, so the copy is what
callers (e.g. :LOGIT-CALLBACK) must hold onto."
  (declare (type fixnum n-vocab))
  (let ((ptr (%llama:get-logits-ith ctx-ptr -1))
        (copy (make-array n-vocab :element-type 'single-float)))
    (declare (type (simple-array single-float (*)) copy))
    (dotimes (i n-vocab)
      (declare (type fixnum i))
      (setf (aref copy i) (cffi:mem-aref ptr :float i)))
    copy))

(defun %invoke-logit-callback (logit-callback ctx-ptr n-vocab)
  "Copy the current logits and invoke LOGIT-CALLBACK with (logits n-vocab).
Returns :ABORT if the callback signaled an error that was not resolved by
the IGNORE-LOGIT-CALLBACK-ERROR restart; returns NIL otherwise."
  (let ((logits (%copy-logits ctx-ptr n-vocab)))
    (restart-case
        (progn
          (handler-bind
              ((error (lambda (c)
                        (declare (ignore c))
                        (invoke-restart 'abort-logit-callback))))
            (funcall logit-callback logits n-vocab))
          nil)
      (ignore-logit-callback-error ()
        :report "Ignore the logit-callback error and continue generation"
        nil)
      (abort-logit-callback ()
        :report "Abort generation due to logit-callback error"
        :abort))))

(defun %grammar-first-sample (chain-ptr grmr-ptr ctx-ptr n-vocab idx)
  "Sample with grammar-first resampling: apply grammar to logits, then chain.
If grammar rejects all candidates, fall back to unconstrained chain sampling.
Returns the selected token ID. Advances the grammar FSM via sampler-accept."
  (declare (optimize (speed 3))
           (type fixnum n-vocab idx))
  (let ((token-size (cffi:foreign-type-size '(:struct %llama:token-data))))
    (declare (type fixnum token-size))
    (cffi:with-foreign-objects ((data-buf '(:struct %llama:token-data) n-vocab)
                                (arr-buf '(:struct %llama:token-data-array)))
      (let ((logits-ptr (%llama:get-logits-ith ctx-ptr idx)))
        (dotimes (i n-vocab)
          (declare (type fixnum i))
          (let ((ptr (cffi:inc-pointer data-buf (* i token-size))))
            (setf (cffi:foreign-slot-value ptr '(:struct %llama:token-data) '%llama:id) i
                  (cffi:foreign-slot-value ptr '(:struct %llama:token-data) '%llama:logit)
                  (cffi:mem-aref logits-ptr :float i)
                  (cffi:foreign-slot-value ptr '(:struct %llama:token-data) '%llama:p) 0.0)))
      (setf (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array) '%llama:data)
            data-buf
            (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array) '%llama:size)
            n-vocab
            (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array) '%llama:selected)
            -1
            (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array) '%llama:sorted)
            0)
      (%llama:sampler-apply grmr-ptr arr-buf)
      (%llama:sampler-apply chain-ptr arr-buf)
      (let* ((selected (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array)
                                                '%llama:selected))
             (id (if (>= selected 0)
                     (let ((ptr (cffi:inc-pointer data-buf (* selected token-size))))
                       (cffi:foreign-slot-value ptr '(:struct %llama:token-data) '%llama:id))
                     ;; Grammar rejected everything — resample without grammar
                     (progn
                       (dotimes (i n-vocab)
                         (declare (type fixnum i))
                         (let ((ptr (cffi:inc-pointer data-buf (* i token-size))))
                           (setf (cffi:foreign-slot-value ptr '(:struct %llama:token-data) '%llama:id) i
                                 (cffi:foreign-slot-value ptr '(:struct %llama:token-data) '%llama:logit)
                                 (cffi:mem-aref logits-ptr :float i)
                                 (cffi:foreign-slot-value ptr '(:struct %llama:token-data) '%llama:p) 0.0)))
                       (setf (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array) '%llama:size)
                             n-vocab
                             (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array) '%llama:selected)
                             -1
                             (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array) '%llama:sorted)
                             0)
                       (%llama:sampler-apply chain-ptr arr-buf)
                       (let* ((sel2 (cffi:foreign-slot-value arr-buf '(:struct %llama:token-data-array)
                                                             '%llama:selected))
                              (ptr2 (cffi:inc-pointer data-buf (* sel2 token-size))))
                         (cffi:foreign-slot-value ptr2 '(:struct %llama:token-data) '%llama:id))))))
        (declare (type fixnum id))
        (%llama:sampler-accept grmr-ptr id)
        (%llama:sampler-accept chain-ptr id)
        id)))))

(llama-defun generate (ctx prompt &rest all-kwargs
                             &key sampler-config
                                  (max-tokens 256) (temp 0.8)
                                  top-k top-p min-p (seed 42)
                                  (parse-special t) prompt-tokens
                                  token-callback
                                  logit-callback
                                  grammar (grammar-root "root")
                                  grammar-first
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
                                  sampler
                                  (reset-context t)
                                  speculative-fns)
  "Generate text by continuing PROMPT. Returns three values: the generated
string, a stop reason (:eog, :length, or :callback), and a (simple-array
fixnum (*)) of the sampled token ids (useful for callers that need to track
the exact KV-cache contents without a lossy re-tokenise round-trip).
Uses the context's model for tokenization. Blocks until EOS or MAX-TOKENS.
Supports extended sampler keywords: :TYPICAL-P, :XTC-PROBABILITY, :XTC-THRESHOLD,
:MIROSTAT, :MIROSTAT-V2, :REPEAT-PENALTY, :FREQUENCY-PENALTY, :PRESENCE-PENALTY,
:DRY-MULTIPLIER, :LOGIT-BIAS, :TOP-N-SIGMA, :DYNAMIC-TEMP-RANGE, :ADAPTIVE-P, etc.

SEED may be an integer (deterministic), :RANDOM (nondeterministic — the C layer
draws a fresh seed internally), or NIL (same as :RANDOM).  Default is 42.  When
:RANDOM is used, the exact seed chosen by the C layer is not observable to the
caller.

When :SAMPLER is provided (a LLAMA-SAMPLER handle, typically from WITH-SAMPLER-CHAIN),
GENERATE borrows the chain and does not free it — the caller owns the lifetime.
All other sampler-related keywords are ignored when :SAMPLER is supplied.

When :RESET-CONTEXT is NIL, the KV cache is NOT cleared before decoding the
prompt.  The caller is responsible for ensuring that the context is already in a
consistent state (e.g. the current cache is a prefix of the new prompt).  The
default (T) preserves the original behaviour: clear the cache and re-prefill
the full prompt each call.

:LOGIT-CALLBACK, when supplied, is called once per token sampled via this
function's own single-token-decode path (the initial token and every
subsequent ordinary token) with (LOGITS N-VOCAB) — LOGITS a fresh
(simple-array single-float (*)) of the raw pre-sampler-chain logits for
the token about to be sampled. Tokens accepted inside a :SPECULATIVE-FNS
draft-verification pass do not trigger it. An error signaled by the
callback aborts generation with STOP-REASON :ERROR unless the
IGNORE-LOGIT-CALLBACK-ERROR restart is invoked.

Signals INPUT-VALIDATION-ERROR if MAX-TOKENS is not a positive integer or
PROMPT is neither a string nor a vector."
  (declare (optimize (speed 3)))
  ;; When :SAMPLER-CONFIG is provided (and no pre-built :SAMPLER chain),
  ;; merge it as defaults for sampler parameters.  Caller-supplied kwargs
  ;; appear first in the appended plist so they win over the config.
  (when (and sampler-config (not sampler))
    (let* ((skip '(:sampler-config :max-tokens :parse-special
                   :prompt-tokens :token-callback :logit-callback
                   :sampler :reset-context :speculative-fns))
           (caller-sampler (loop for (k v) on all-kwargs by #'cddr
                                 unless (member k skip)
                                 nconc (list k v)))
           (effective (append caller-sampler sampler-config)))
      (setf temp              (getf effective :temp 0.8)
            top-k             (getf effective :top-k nil)
            top-p             (getf effective :top-p nil)
            min-p             (getf effective :min-p nil)
            seed              (getf effective :seed 42)
            grammar           (getf effective :grammar nil)
            grammar-root      (getf effective :grammar-root "root")
            grammar-first     (getf effective :grammar-first nil)
            typical-p         (getf effective :typical-p nil)
            xtc-probability   (getf effective :xtc-probability nil)
            xtc-threshold     (getf effective :xtc-threshold nil)
            top-n-sigma       (getf effective :top-n-sigma nil)
            mirostat          (getf effective :mirostat nil)
            mirostat-v2       (getf effective :mirostat-v2 nil)
            mirostat-tau      (getf effective :mirostat-tau 5.0)
            mirostat-eta      (getf effective :mirostat-eta 0.1)
            repeat-penalty    (getf effective :repeat-penalty nil)
            frequency-penalty (getf effective :frequency-penalty nil)
            presence-penalty  (getf effective :presence-penalty nil)
            penalty-last-n    (getf effective :penalty-last-n 64)
            dry-multiplier    (getf effective :dry-multiplier nil)
            dry-base          (getf effective :dry-base 1.75)
            dry-allowed-length    (getf effective :dry-allowed-length 2)
            dry-penalty-last-n    (getf effective :dry-penalty-last-n -1)
            dry-seq-breakers  (getf effective :dry-seq-breakers nil)
            logit-bias        (getf effective :logit-bias nil)
            dynamic-temp-range    (getf effective :dynamic-temp-range nil)
            dynamic-temp-exponent (getf effective :dynamic-temp-exponent 1.0)
            adaptive-p        (getf effective :adaptive-p nil)
            adaptive-p-decay  (getf effective :adaptive-p-decay 0.0))))
  (check-type max-tokens (integer 1 *))
  (unless prompt-tokens
    (check-type prompt (or string vector)))
  (let* ((ctx-ptr (llama-context-pointer ctx))
         (raw-model (%llama:get-model ctx-ptr))
         (model (%make-llama-model :pointer raw-model))
         (vocab (%llama:model-get-vocab raw-model))
         (n-vocab (%llama:vocab-n-tokens vocab))
         (prompt-tokens (or prompt-tokens
                            (tokenize model prompt :parse-special parse-special)))
         (n-prompt (length prompt-tokens))
         (generated (make-array 0 :element-type 'fixnum
                                  :adjustable t :fill-pointer 0)))
    (declare (type fixnum n-prompt n-vocab)
             (type (vector fixnum) generated))
    (when reset-context
      (%llama:memory-clear (%llama:get-memory ctx-ptr) 1))
    ;; Decode the prompt via prefill (skip if empty — caller already prefilled)
    (when (plusp n-prompt)
      (prefill ctx prompt-tokens))
    ;; Speculative decoding setup
    (let* ((begin-fn  (getf speculative-fns :begin-fn))
           (draft-fn  (getf speculative-fns :draft-fn))
           (accept-fn (getf speculative-fns :accept-fn))
           (all-tokens (when draft-fn
                         (let ((v (make-array n-prompt :element-type 'fixnum
                                                       :adjustable t
                                                       :fill-pointer n-prompt)))
                           (dotimes (i n-prompt)
                             (setf (aref v i) (aref prompt-tokens i)))
                           v))))
      (when (and begin-fn (plusp n-prompt))
        (funcall begin-fn 0 prompt-tokens))
      ;; Warn if caller supplied a chain but also passed sampler-building kwargs
    (when (and sampler
               (or sampler-config
                   grammar top-k top-p min-p typical-p xtc-probability
                   dry-multiplier logit-bias mirostat mirostat-v2
                   repeat-penalty frequency-penalty presence-penalty
                   adaptive-p dynamic-temp-range top-n-sigma))
      (warn "~@<When :SAMPLER is provided, other sampler keywords ~
(:SAMPLER-CONFIG, :GRAMMAR, :TOP-K, :TEMP, etc.) are ignored.~@:>"))
    ;; Generation loop
    (multiple-value-bind (built-chain-ptr built-grmr-ptr)
        (if sampler
            (values (llama-sampler-pointer sampler) nil)
            (build-sampler-chain
             :temp temp :top-k top-k :top-p top-p
             :min-p min-p :seed seed
             :model model :grammar grammar
             :grammar-root grammar-root
             :grammar-first grammar-first
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
    (let ((chain-ptr built-chain-ptr)
          (grmr-ptr built-grmr-ptr)
          (emitted-len 0)
          (stop-reason nil))
      (declare (type fixnum emitted-len))
      (unwind-protect
          (let ((n-past (the fixnum
                           (if (plusp n-prompt)
                               n-prompt
                               (1+ (nth-value 1 (kv-cache-pos ctx 0)))))))
            (declare (type fixnum n-past))
            ;; When grmr-ptr is nil, sampler-sample calls sampler-accept
            ;; internally — do NOT call sampler-accept again.
            ;; When grmr-ptr is non-nil, %grammar-first-sample handles
            ;; both grammar application and accept.
            (macrolet ((sample ()
                         '(progn
                            (when (and logit-callback
                                       (eq :abort (%invoke-logit-callback
                                                   logit-callback ctx-ptr n-vocab)))
                              (setf stop-reason :error)
                              (return))
                            (if grmr-ptr
                                (%grammar-first-sample
                                 chain-ptr grmr-ptr ctx-ptr n-vocab -1)
                                (%llama:sampler-sample chain-ptr ctx-ptr -1)))))
              (loop with sampled of-type fixnum = (sample)
                    while (< (length generated) max-tokens)
                    do
                 ;; 1. EOG check
                 (unless (zerop (%llama:token-is-eog vocab sampled))
                   (setf stop-reason :eog)
                   (return))
                 ;; 2. Record token (all-tokens updated after speculative branch
                 ;;    so draft-fn sees history WITHOUT the current sampled)
                 (vector-push-extend sampled generated)
                 (when token-callback
                   (let* ((full (detokenize model generated :remove-special t))
                          (new-text (subseq full emitted-len)))
                     (when (plusp (length new-text))
                       (setf emitted-len (length full))
                       (restart-case
                           (handler-bind
                               ((error (lambda (c)
                                         (declare (ignore c))
                                         (invoke-restart 'abort-generation))))
                             (unless (funcall token-callback new-text)
                               (setf stop-reason :callback)
                               (return)))
                         (ignore-callback-error ()
                           :report "Ignore the callback error and continue generation"
                           nil)
                         (abort-generation ()
                           :report "Abort generation due to token-callback error"
                           (setf stop-reason :error)
                           (return))))))
                 (when (or stop-reason (>= (length generated) max-tokens))
                   (return))
                 ;; 3. Speculative branch
                 (if draft-fn
                     (let* ((drafts (funcall draft-fn
                                             :seq-id 0 :n-past n-past
                                             :id-last sampled
                                             :prompt-tokens all-tokens))
                            (k (length drafts)))
                       (if (plusp k)
                           ;; Batch decode: sampled + K drafts
                           (progn
                             (with-batch (batch (1+ k))
                               (batch-add-token batch sampled n-past 0 :logits t)
                               (dotimes (i k)
                                 (batch-add-token batch (aref drafts i)
                                                  (+ n-past 1 i) 0 :logits t))
                               (batch-decode ctx batch))
                             (let ((n-accepted 0)
                                   (mismatch-token nil))
                               (declare (type fixnum n-accepted))
                               ;; Verify each draft
                               (block verify
                                 (dotimes (i k)
                                   (let ((target (%llama:sampler-sample
                                                  chain-ptr ctx-ptr i)))
                                     (if (= target (aref drafts i))
                                         (progn
                                           (incf n-accepted)
                                           ;; Check EOG on accepted draft
                                           (unless (zerop (%llama:token-is-eog
                                                           vocab (aref drafts i)))
                                             (setf stop-reason :eog)
                                             (return-from verify))
                                           ;; Record accepted draft
                                           (vector-push-extend (aref drafts i)
                                                               generated)
                                           (when all-tokens
                                             (vector-push-extend (aref drafts i)
                                                                 all-tokens))
                                           (when token-callback
                                             (let* ((full (detokenize model generated
                                                                      :remove-special t))
                                                    (new-text (subseq full emitted-len)))
                                               (when (plusp (length new-text))
                                                 (setf emitted-len (length full))
                                                 (restart-case
                                                     (handler-bind
                                                         ((error (lambda (c)
                                                                   (declare (ignore c))
                                                                   (invoke-restart
                                                                    'abort-generation))))
                                                       (unless (funcall token-callback
                                                                        new-text)
                                                         (setf stop-reason :callback)
                                                         (return-from verify)))
                                                   (ignore-callback-error ()
                                                     :report "Ignore callback error"
                                                     nil)
                                                   (abort-generation ()
                                                     :report "Abort generation"
                                                     (setf stop-reason :error)
                                                     (return-from verify))))))
                                           (when (or stop-reason
                                                     (>= (length generated)
                                                         max-tokens))
                                             (return-from verify)))
                                         ;; Mismatch — save target's token, stop verification
                                         (progn
                                           (setf mismatch-token target)
                                           (return-from verify))))))
                               ;; Record sampled in all-tokens now that drafting is done
                               (when all-tokens
                                 (vector-push-extend sampled all-tokens))
                               ;; Accept/evict
                               (when accept-fn
                                 (funcall accept-fn 0 n-accepted))
                               (incf n-past (1+ n-accepted))
                               (when (< n-accepted k)
                                 (kv-cache-seq-rm ctx 0 n-past -1))
                               ;; Next sample
                               (unless (or stop-reason
                                           (>= (length generated) max-tokens))
                                 (setf sampled
                                       (if mismatch-token
                                           mismatch-token
                                           ;; All accepted: sample from position after last
                                           (%llama:sampler-sample chain-ptr ctx-ptr k))))))
                           ;; No drafts available — single token decode
                           (progn
                             (when all-tokens
                               (vector-push-extend sampled all-tokens))
                             (cffi:with-foreign-object (tok-buf '%llama:token 1)
                               (setf (cffi:mem-aref tok-buf '%llama:token 0) sampled)
                               (let* ((batch (%llama:batch-get-one tok-buf 1))
                                      (rc (%llama:decode ctx-ptr batch)))
                                 (declare (type fixnum rc))
                                 (unless (zerop rc)
                                   (error 'decode-error :code rc)))
                               (setf (llama-context-compute-pending-p ctx) t))
                             (incf n-past)
                             (setf sampled (sample)))))
                     ;; No speculative fns — original single-token path
                     (progn
                       (cffi:with-foreign-object (tok-buf '%llama:token 1)
                         (setf (cffi:mem-aref tok-buf '%llama:token 0) sampled)
                         (let* ((batch (%llama:batch-get-one tok-buf 1))
                                (rc (%llama:decode ctx-ptr batch)))
                           (declare (type fixnum rc))
                           (unless (zerop rc)
                             (error 'decode-error :code rc)))
                         (setf (llama-context-compute-pending-p ctx) t))
                       (incf n-past)
                       (setf sampled (sample))))
                 (when stop-reason (return)))))
        (unless sampler
          (when grmr-ptr (%llama:sampler-free grmr-ptr))
          (%llama:sampler-free chain-ptr)))
    ;; Convert generated tokens to string.  Always materialise result-tokens so
    ;; it can be returned as the third value for callers that need exact cache
    ;; contents (e.g. CHAT-SESSION-SEND) without a lossy re-tokenise.
    (let* ((n-gen (length generated))
           (result-tokens (make-array n-gen :element-type 'fixnum))
           (text (if (zerop n-gen)
                     ""
                     (progn
                       (dotimes (i n-gen)
                         (declare (type fixnum i))
                         (setf (aref result-tokens i) (aref generated i)))
                       (detokenize model result-tokens :remove-special t)))))
      (declare (type fixnum n-gen)
               (type (simple-array fixnum (*)) result-tokens))
      (values text
              (or stop-reason
                  (if (= n-gen max-tokens) :length :eog))
              result-tokens)))))))

(llama-defun embed (ctx text &key (normalize t))
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
        result))))
