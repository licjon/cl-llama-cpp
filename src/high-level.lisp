(in-package #:cl-llama-cpp)

(defvar *backend-initialized* nil)

(defun ensure-backend ()
  (unless *backend-initialized*
    (with-fp-traps-masked
      (%llama:backend-init))
    (setf *backend-initialized* t)))

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
  (let ((model-ptr (gensym "MODEL")))
    `(progn
       (ensure-backend)
       (with-fp-traps-masked
         (let* ((defaults (%llama:model-default-params))
                (model-params (override-params defaults (list ,@params)))
                (,model-ptr (%llama:model-load-from-file ,path model-params)))
           (when (cffi:null-pointer-p ,model-ptr)
             (error 'model-load-error :path ,path))
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
             (%llama:free ,var))))))

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
          ;; Second pass: fill text buffer
          (cffi:with-foreign-pointer-as-string (text-buf (1+ n-needed))
            (%llama:detokenize vocab tok-buf n-tokens
                               text-buf n-needed
                               remove-sp unparse-sp))))))))
