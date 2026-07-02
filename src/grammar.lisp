(in-package #:cl-llama-cpp)

;;; Grammar / constrained generation wrappers

(llama-defun make-grammar-sampler (model grammar &key (root "root"))
  "Create a grammar sampler from a GBNF grammar string and root rule.
Returns a LLAMA-SAMPLER handle. Caller must free with
(%llama:sampler-free (llama-sampler-pointer s)), or add to a sampler chain."
  (check-type grammar string)
  (restart-case
      (progn
        (when (zerop (length grammar))
          (error 'grammar-error :grammar grammar))
        (let* ((vocab (%llama:model-get-vocab (llama-model-pointer model)))
               (sampler (%llama:sampler-init-grammar vocab grammar root)))
          (when (cffi:null-pointer-p sampler)
            (error 'grammar-error :grammar grammar))
          (%make-llama-sampler :pointer sampler)))
    (skip-grammar ()
      :report "Continue without grammar constraint"
      nil)
    (use-different-grammar (g)
      :report "Retry with a different grammar string"
      :interactive (lambda ()
                     (format *query-io* "Grammar: ")
                     (list (read-line *query-io*)))
      (make-grammar-sampler model g :root root))))

(llama-defun make-grammar-sampler-lazy (model grammar &key (root "root")
                                                      trigger-words
                                                      trigger-patterns
                                                      trigger-tokens)
  "Create a lazy grammar sampler that activates only when triggered.
When TRIGGER-PATTERNS is provided, uses pattern matching; otherwise uses
TRIGGER-WORDS for exact word matching. TRIGGER-TOKENS are token IDs that
also trigger grammar activation.
Returns a LLAMA-SAMPLER handle. Caller must free with
(%llama:sampler-free (llama-sampler-pointer s))."
  (check-type grammar string)
  (when (and trigger-words trigger-patterns)
    (error "Cannot specify both :TRIGGER-WORDS and :TRIGGER-PATTERNS"))
  (restart-case
      (progn
        (when (zerop (length grammar))
          (error 'grammar-error :grammar grammar))
        (let* ((vocab (%llama:model-get-vocab (llama-model-pointer model)))
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
                  (%make-llama-sampler :pointer sampler)))
            (dolist (ptr foreign-strings)
              (cffi:foreign-string-free ptr)))))
    (skip-grammar ()
      :report "Continue without grammar constraint"
      nil)
    (use-different-grammar (g)
      :report "Retry with a different grammar string"
      :interactive (lambda ()
                     (format *query-io* "Grammar: ")
                     (list (read-line *query-io*)))
      (make-grammar-sampler-lazy model g :root root
                                  :trigger-words trigger-words
                                  :trigger-patterns trigger-patterns
                                  :trigger-tokens trigger-tokens))))

(llama-defun make-infill-sampler (model)
  "Create a fill-in-the-middle sampler for FIM-capable models.
Returns a LLAMA-SAMPLER handle. Caller must free with
(%llama:sampler-free (llama-sampler-pointer s))."
  (restart-case
      (let* ((vocab (%llama:model-get-vocab (llama-model-pointer model)))
             (sampler (%llama:sampler-init-infill vocab)))
        (when (cffi:null-pointer-p sampler)
          (error 'grammar-error :grammar "<infill>"))
        (%make-llama-sampler :pointer sampler))
    (skip-grammar ()
      :report "Continue without infill grammar constraint"
      nil)))

(defmacro with-grammar-sampler ((var model grammar &key (root "root") lazy
                                                         trigger-words
                                                         trigger-patterns
                                                         trigger-tokens)
                                &body body)
  "Create a grammar sampler, bind to VAR, execute BODY, free the sampler.
When LAZY is true, creates a lazy grammar sampler with optional trigger args."
  (let ((sampler-ptr (gensym "GRAMMAR-SAMPLER")))
    `(with-llama-compatible-fp-environment
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
           (when ,sampler-ptr
             (%llama:sampler-free (llama-sampler-pointer ,sampler-ptr))))))))

