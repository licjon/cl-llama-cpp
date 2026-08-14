(in-package #:cl-llama-cpp/common/speculative)

;;; Type constants (match common_speculative_type enum values)

(defconstant +speculative-type-none+         0)
(defconstant +speculative-type-draft-simple+ 1)
(defconstant +speculative-type-draft-eagle3+ 2)
(defconstant +speculative-type-draft-mtp+    3)
(defconstant +speculative-type-ngram-simple+ 4)
(defconstant +speculative-type-ngram-map-k+  5)
(defconstant +speculative-type-ngram-map-k4v+ 6)
(defconstant +speculative-type-ngram-mod+    7)
(defconstant +speculative-type-ngram-cache+  8)

;;; Conditions

(define-condition speculative-init-error (cl-llama-cpp:llama-error)
  ((message :initarg :message :reader speculative-init-error-message))
  (:report (lambda (c s)
             (format s "Speculative decoding init failed: ~A"
                     (speculative-init-error-message c)))))

;;; Params builder

(defstruct (speculative-params
             (:constructor %make-speculative-params)
             (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer))

(defun make-speculative-params ()
  "Create a speculative decoding params builder. Must be freed with
FREE-SPECULATIVE-PARAMS or used within WITH-SPECULATIVE-PARAMS."
  (let ((ptr (%params-create)))
    (when (cffi:null-pointer-p ptr)
      (error 'speculative-init-error :message "Failed to allocate params"))
    (%make-speculative-params :pointer ptr)))

(defun free-speculative-params (params)
  "Free a speculative params builder. Idempotent."
  (let ((ptr (speculative-params-pointer params)))
    (unless (cffi:null-pointer-p ptr)
      (%params-free ptr)
      (setf (speculative-params-pointer params) (cffi:null-pointer))))
  nil)

(defmacro with-speculative-params ((var) &body body)
  "Create a speculative params builder, bind to VAR, execute BODY, free."
  `(let ((,var (make-speculative-params)))
     (unwind-protect (progn ,@body)
       (free-speculative-params ,var))))

(defun speculative-params-add-type (params type)
  "Add a speculative decoding type to PARAMS. TYPE is one of the
+SPECULATIVE-TYPE-*+ constants. Can be called multiple times."
  (%params-add-type (speculative-params-pointer params) type))

(defun speculative-params-set-draft-n-max (params n)
  (%params-set-draft-n-max (speculative-params-pointer params) n))

(defun speculative-params-set-draft-n-min (params n)
  (%params-set-draft-n-min (speculative-params-pointer params) n))

(defun speculative-params-set-draft-p-min (params val)
  (%params-set-draft-p-min (speculative-params-pointer params)
                           (coerce val 'single-float)))

(defun speculative-params-set-draft-p-split (params val)
  (%params-set-draft-p-split (speculative-params-pointer params)
                             (coerce val 'single-float)))

(defun speculative-params-set-ngram-n (params v)
  (%params-set-ngram-n (speculative-params-pointer params) v))

(defun speculative-params-set-ngram-m (params v)
  (%params-set-ngram-m (speculative-params-pointer params) v))

(defun speculative-params-set-ngram-min-hits (params v)
  (%params-set-ngram-min-hits (speculative-params-pointer params) v))

(defun speculative-params-n-max (params)
  "Return the maximum number of draft tokens for the given params."
  (%spec-n-max (speculative-params-pointer params)))

;;; Speculative context

(defstruct (speculative-context
             (:constructor %make-speculative-context)
             (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer)
  (n-seq 1 :type (unsigned-byte 32))
  (freed-cell (list nil) :type cons :read-only t))

(defun %try-claim-for-free (cell)
  #+sbcl (null (sb-ext:cas (car cell) nil t))
  #-sbcl (prog1 (null (car cell))
           (setf (car cell) t)))

(defun make-speculative-context (params &key (n-seq 1))
  "Initialize a speculative decoding context from PARAMS.
Must be freed with FREE-SPECULATIVE-CONTEXT or used within
WITH-SPECULATIVE-CONTEXT."
  (let ((result (%spec-init (speculative-params-pointer params) n-seq)))
    (let ((ctx-ptr (getf result 'ctx))
          (err-ptr (getf result 'error)))
      (unwind-protect
           (cond
             ((and (not (cffi:null-pointer-p ctx-ptr))
                   (cffi:null-pointer-p err-ptr))
              (let ((ctx (%make-speculative-context
                          :pointer ctx-ptr :n-seq n-seq)))
                (%register-spec-finalizer ctx)
                ctx))
             ((not (cffi:null-pointer-p err-ptr))
              (error 'speculative-init-error
                     :message (cffi:foreign-string-to-lisp err-ptr)))
             (t
              (error 'speculative-init-error
                     :message "init returned NULL without error message")))
        (unless (cffi:null-pointer-p err-ptr)
          (%shim-free err-ptr))))))

(defun free-speculative-context (ctx)
  "Free a speculative decoding context. Idempotent."
  (when (%try-claim-for-free (speculative-context-freed-cell ctx))
    (tg:cancel-finalization ctx)
    (%spec-free (speculative-context-pointer ctx)))
  nil)

(defun %register-spec-finalizer (ctx)
  (let ((ptr (speculative-context-pointer ctx))
        (cell (speculative-context-freed-cell ctx)))
    (tg:finalize ctx
      (lambda ()
        (when (%try-claim-for-free cell)
          (ignore-errors (%spec-free ptr)))))))

(defmacro with-speculative-context ((var params &key (n-seq 1)) &body body)
  "Initialize a speculative context, bind to VAR, execute BODY, free."
  `(let ((,var (make-speculative-context ,params :n-seq ,n-seq)))
     (unwind-protect (progn ,@body)
       (free-speculative-context ,var))))

;;; Operations

(defun speculative-begin (ctx seq-id prompt-tokens)
  "Begin a new speculative decoding generation for SEQ-ID.
PROMPT-TOKENS is a vector/list of token IDs (fixnums)."
  (let ((tokens (coerce prompt-tokens 'vector)))
    (cffi:with-foreign-object (buf :int32 (length tokens))
      (loop for i below (length tokens)
            do (setf (cffi:mem-aref buf :int32 i) (aref tokens i)))
      (%spec-begin (speculative-context-pointer ctx)
                   seq-id buf (length tokens)))))

(defun speculative-draft (ctx &key (seq-id 0) n-past id-last
                                   prompt-tokens (n-max -1)
                                   (max-draft-tokens 64))
  "Draft speculative tokens for SEQ-ID.
N-PAST is the number of tokens already decoded.
ID-LAST is the most recently accepted token.
PROMPT-TOKENS is a vector/list of the prompt token IDs.
Returns a vector of drafted token IDs."
  (let ((ptr (speculative-context-pointer ctx)))
    (%dp-set-drafting ptr seq-id 1)
    (when n-past (%dp-set-n-past ptr seq-id n-past))
    (when id-last (%dp-set-id-last ptr seq-id id-last))
    (%dp-set-n-max ptr seq-id n-max)
    (when prompt-tokens
      (let ((tokens (coerce prompt-tokens 'vector)))
        (cffi:with-foreign-object (buf :int32 (length tokens))
          (loop for i below (length tokens)
                do (setf (cffi:mem-aref buf :int32 i) (aref tokens i)))
          (%dp-set-prompt ptr seq-id buf (length tokens)))))
    (%dp-prepare-result ptr seq-id)
    (%spec-draft ptr)
    (cffi:with-foreign-object (out :int32 max-draft-tokens)
      (let ((n (%dp-get-result ptr seq-id out max-draft-tokens)))
        (let ((result (make-array n :element-type 'fixnum)))
          (loop for i below n
                do (setf (aref result i) (cffi:mem-aref out :int32 i)))
          result)))))

(defun speculative-accept (ctx seq-id n-accepted)
  "Report that N-ACCEPTED draft tokens were accepted for SEQ-ID."
  (%spec-accept (speculative-context-pointer ctx) seq-id n-accepted))

(defun speculative-need-embd-p (ctx)
  "Return T if any speculative implementation needs target embeddings."
  (not (zerop (%spec-need-embd (speculative-context-pointer ctx)))))

(defun speculative-need-embd-nextn-p (ctx)
  "Return T if any speculative implementation needs nextn embeddings."
  (not (zerop (%spec-need-embd-nextn (speculative-context-pointer ctx)))))

(defun speculative-print-stats (ctx)
  "Print speculative decoding statistics to *error-output*."
  (%spec-print-stats (speculative-context-pointer ctx)))
