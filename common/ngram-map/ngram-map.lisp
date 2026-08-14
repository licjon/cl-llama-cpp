(in-package #:cl-llama-cpp/common/ngram-map)

;;; Conditions

(define-condition ngram-map-init-error (cl-llama-cpp:llama-error)
  ((message :initarg :message :reader ngram-map-init-error-message))
  (:report (lambda (c s)
             (format s "N-gram map init failed: ~A"
                     (ngram-map-init-error-message c)))))

;;; Token marshaling

(defun %marshal-tokens (token-seq body-fn)
  (let ((tokens (coerce token-seq 'vector)))
    (let ((n (length tokens)))
      (if (zerop n)
          (funcall body-fn (cffi:null-pointer) 0)
          (cffi:with-foreign-object (buf :int32 n)
            (loop for i below n
                  do (setf (cffi:mem-aref buf :int32 i) (aref tokens i)))
            (funcall body-fn buf n))))))

(defun %read-token-buf (buf n)
  (let ((result (make-array n :element-type 'fixnum)))
    (loop for i below n
          do (setf (aref result i) (cffi:mem-aref buf :int32 i)))
    result))

;;; Simple n-gram draft

(defun ngram-simple-draft (size-ngram size-mgram tokens sampled
                           &key (max-draft-tokens 64))
  "Draft tokens using simple n-gram pattern matching in token history.
SIZE-NGRAM is the n-gram size to look up. SIZE-MGRAM is the m-gram size to draft.
TOKENS is a sequence of token IDs (the generation history).
SAMPLED is the most recently sampled token.
Returns a vector of drafted token IDs (empty if no match found)."
  (%marshal-tokens tokens
    (lambda (tok-buf n-tok)
      (cffi:with-foreign-object (out :int32 max-draft-tokens)
        (let ((n (%ngram-simple-draft
                  size-ngram size-mgram
                  tok-buf n-tok sampled
                  out max-draft-tokens)))
          (%read-token-buf out n))))))

;;; Map lifecycle

(defstruct (ngram-map
             (:constructor %make-ngram-map)
             (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer)
  (freed-cell (list nil) :type cons :read-only t))

(defun %try-claim-for-free (cell)
  #+sbcl (null (sb-ext:cas (car cell) nil t))
  #-sbcl (prog1 (null (car cell))
           (setf (car cell) t)))

(defun make-ngram-map (size-key size-value &key key-only-p (min-hits 1))
  "Create an n-gram map for speculative token drafting.
SIZE-KEY is the key n-gram size. SIZE-VALUE is the value m-gram size.
KEY-ONLY-P when true uses only key lookups (simpler, no value tracking).
MIN-HITS is the minimum key occurrences before drafting.
Must be freed with FREE-NGRAM-MAP or used within WITH-NGRAM-MAP."
  (let ((ptr (%ngram-map-create size-key size-value
                                (if key-only-p 1 0) min-hits)))
    (when (cffi:null-pointer-p ptr)
      (error 'ngram-map-init-error :message "Failed to allocate ngram map"))
    (let ((m (%make-ngram-map :pointer ptr)))
      (%register-ngram-map-finalizer m)
      m)))

(defun free-ngram-map (map)
  "Free an n-gram map. Idempotent."
  (when (%try-claim-for-free (ngram-map-freed-cell map))
    (tg:cancel-finalization map)
    (%ngram-map-free (ngram-map-pointer map)))
  nil)

(defun %register-ngram-map-finalizer (map)
  (let ((ptr (ngram-map-pointer map))
        (cell (ngram-map-freed-cell map)))
    (tg:finalize map
      (lambda ()
        (when (%try-claim-for-free cell)
          (ignore-errors (%ngram-map-free ptr)))))))

(defmacro with-ngram-map ((var size-key size-value
                            &key key-only-p (min-hits 1))
                           &body body)
  "Create an n-gram map, bind to VAR, execute BODY, free."
  `(let ((,var (make-ngram-map ,size-key ,size-value
                               :key-only-p ,key-only-p
                               :min-hits ,min-hits)))
     (unwind-protect (progn ,@body)
       (free-ngram-map ,var))))

;;; Map operations

(defun ngram-map-begin (map tokens)
  "Initialize or refresh the n-gram map with current token history.
Must be called before NGRAM-MAP-DRAFT, and again after context truncation."
  (%marshal-tokens tokens
    (lambda (tok-buf n-tok)
      (%ngram-map-begin (ngram-map-pointer map) tok-buf n-tok)))
  nil)

(defun ngram-map-draft (map tokens sampled &key (max-draft-tokens 64))
  "Draft tokens using hash-map n-gram lookup.
MAP is an ngram-map created with MAKE-NGRAM-MAP.
TOKENS is the generation history. SAMPLED is the last sampled token.
Returns a vector of drafted token IDs (empty if no match)."
  (%marshal-tokens tokens
    (lambda (tok-buf n-tok)
      (cffi:with-foreign-object (out :int32 max-draft-tokens)
        (let ((n (%ngram-map-draft
                  (ngram-map-pointer map)
                  tok-buf n-tok sampled
                  out max-draft-tokens)))
          (%read-token-buf out n))))))

(defun ngram-map-accept (map n-accepted)
  "Update map statistics with the number of accepted draft tokens.
Call after verifying draft tokens against the target model."
  (%ngram-map-accept (ngram-map-pointer map) n-accepted)
  nil)
