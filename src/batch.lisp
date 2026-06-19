(in-package #:cl-llama-cpp)

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
    `(with-llama-compatible-fp-environment
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
  (with-llama-compatible-fp-environment
    (let ((rc (%llama:decode ctx (%batch-data batch))))
      (unless (zerop rc)
        (error 'decode-error :code rc))
      nil)))

(defun batch-encode (ctx batch)
  "Encode BATCH using context CTX.
Signals DECODE-ERROR on failure. Returns NIL on success."
  (with-llama-compatible-fp-environment
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
  (with-llama-compatible-fp-environment
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
