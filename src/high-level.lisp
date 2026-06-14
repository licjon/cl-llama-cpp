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
                               text-buf (1+ n-needed)
                               remove-sp unparse-sp))))))))

(defun build-sampler-chain (&key (temp 0.8) top-k top-p min-p (seed 42) greedy)
  "Build and return a sampler chain pointer. Caller must free with %llama:sampler-free."
  (let ((chain (%llama:sampler-chain-init
                (%llama:sampler-chain-default-params))))
    (cond
      (greedy
       (%llama:sampler-chain-add chain (%llama:sampler-init-greedy)))
      (t
       (when top-k
         (%llama:sampler-chain-add chain (%llama:sampler-init-top-k top-k)))
       (when top-p
         (%llama:sampler-chain-add chain (%llama:sampler-init-top-p (coerce top-p 'single-float) 1)))
       (when min-p
         (%llama:sampler-chain-add chain (%llama:sampler-init-min-p (coerce min-p 'single-float) 1)))
       (%llama:sampler-chain-add chain (%llama:sampler-init-temp (coerce temp 'single-float)))
       (%llama:sampler-chain-add chain (%llama:sampler-init-dist seed))))
    chain))

(defmacro with-sampler-chain ((var &rest args
                                &key (temp 0.8) top-k top-p min-p
                                     (seed 42) greedy) &body body)
  "Create a sampler chain, bind to VAR, execute BODY, free the chain."
  (declare (ignore temp top-k top-p min-p seed greedy))
  (let ((chain (gensym "CHAIN")))
    `(with-fp-traps-masked
       (let ((,chain (build-sampler-chain ,@args)))
         (unwind-protect
              (let ((,var ,chain))
                ,@body)
           (%llama:sampler-free ,chain))))))

(defun generate (ctx prompt &key (max-tokens 256) (temp 0.8)
                                  top-k top-p min-p (seed 42))
  "Generate text by continuing PROMPT. Returns the generated string.
Uses the context's model for tokenization. Blocks until EOS or MAX-TOKENS."
  (with-fp-traps-masked
    (let* ((model (%llama:get-model ctx))
           (vocab (%llama:model-get-vocab model))
           (prompt-tokens (tokenize model prompt))
           (n-prompt (length prompt-tokens))
           (eos (%llama:token-eos vocab))
           (generated (make-array 0 :element-type 'fixnum
                                    :adjustable t :fill-pointer 0)))
      ;; Decode the prompt
      (cffi:with-foreign-object (tok-buf '%llama:token n-prompt)
        (dotimes (i n-prompt)
          (setf (cffi:mem-aref tok-buf '%llama:token i) (aref prompt-tokens i)))
        (let* ((batch (%llama:batch-get-one tok-buf n-prompt))
               (rc (%llama:decode ctx batch)))
          (unless (zerop rc)
            (error 'decode-error :code rc))))
      ;; Generation loop
      (let ((sampler (build-sampler-chain :temp temp :top-k top-k :top-p top-p
                                          :min-p min-p :seed seed)))
        (unwind-protect
            (loop for i from 0 below max-tokens
                  for new-token = (%llama:sampler-sample sampler ctx -1)
                  do (%llama:sampler-accept sampler new-token)
                  until (= new-token eos)
                  do (vector-push-extend new-token generated)
                     (cffi:with-foreign-object (tok-buf '%llama:token 1)
                       (setf (cffi:mem-aref tok-buf '%llama:token 0) new-token)
                       (let* ((batch (%llama:batch-get-one tok-buf 1))
                              (rc (%llama:decode ctx batch)))
                         (unless (zerop rc)
                           (error 'decode-error :code rc)))))
          (%llama:sampler-free sampler)))
      ;; Convert generated tokens to string
      (if (zerop (length generated))
          ""
          (let ((result-tokens (make-array (length generated) :element-type 'fixnum)))
            (dotimes (i (length generated))
              (setf (aref result-tokens i) (aref generated i)))
            (detokenize model result-tokens))))))

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
