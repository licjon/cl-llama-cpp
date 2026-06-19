(in-package #:cl-llama-cpp)

;;; Grammar / constrained generation wrappers

(defun make-grammar-sampler (model grammar &key (root "root"))
  "Create a grammar sampler from a GBNF grammar string and root rule.
Returns a sampler pointer. Caller must free with %llama:sampler-free,
or add to a sampler chain (which frees it automatically)."
  (check-type grammar string)
  (when (zerop (length grammar))
    (error 'grammar-error :grammar grammar))
  (with-llama-compatible-fp-environment
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
  (with-llama-compatible-fp-environment
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
  (with-llama-compatible-fp-environment
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
           (%llama:sampler-free ,sampler-ptr))))))
