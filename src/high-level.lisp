(in-package #:cl-llama-cpp)

;;; Every %llama symbol the high-level API depends on.  After regenerating
;;; bindings, call (cl-llama-cpp:check-binding-deps) to verify none were
;;; removed upstream.  The generator also reads this list to flag removals.
(defparameter *binding-deps*
  '(;; Lifecycle
    %llama:backend-init
    %llama:model-default-params %llama:model-load-from-file %llama:model-free
    %llama:context-default-params %llama:new-context-with-model %llama:free
    %llama:get-model %llama:get-memory %llama:memory-clear
    ;; Tokenization
    %llama:model-get-vocab %llama:tokenize %llama:detokenize %llama:token
    %llama:token-bos %llama:token-is-eog
    ;; Generation
    %llama:batch-get-one %llama:decode %llama:encode
    %llama:sampler-chain-default-params %llama:sampler-chain-init
    %llama:sampler-chain-add %llama:sampler-sample %llama:sampler-accept
    %llama:sampler-free
    %llama:sampler-init-greedy %llama:sampler-init-temp
    %llama:sampler-init-top-k %llama:sampler-init-top-p %llama:sampler-init-min-p
    %llama:sampler-init-dist
    ;; Embeddings
    %llama:model-n-embd %llama:get-embeddings-ith
    ;; Chat templates
    %llama:chat-apply-template %llama:chat-builtin-templates
    %llama:chat-message %llama:model-chat-template
    ;; LoRA adapters
    %llama:adapter-lora-init %llama:adapter-lora-free
    %llama:set-adapters-lora
    %llama:adapter-meta-val-str %llama:adapter-meta-count
    %llama:adapter-meta-key-by-index %llama:adapter-meta-val-str-by-index
    %llama:adapter-get-alora-n-invocation-tokens
    %llama:adapter-get-alora-invocation-tokens
    ;; Control vectors
    %llama:set-adapter-cvec))

(defun check-binding-deps ()
  "Verify that every symbol in *BINDING-DEPS* is fbound or a known type.
Returns T if all present, signals a warning per missing symbol."
  (let ((missing nil))
    (dolist (sym *binding-deps*)
      (unless (or (fboundp sym)
                  (ignore-errors (cffi:foreign-type-size sym))
                  (ignore-errors (cffi:foreign-type-size `(:struct ,sym))))
        (push sym missing)))
    (if missing
        (progn
          (warn "~D binding~:P missing from %llama after regeneration:~%~{  ~S~%~}"
                (length missing) (nreverse missing))
          nil)
        (progn
          (format t "~&All ~D binding dependencies present.~%" (length *binding-deps*))
          t))))

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
             (%llama:free ,var)))))))

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
          ;; Second pass: fill text buffer (llama API does not null-terminate)
          (cffi:with-foreign-pointer (text-buf (1+ n-needed))
            (let ((n-written (%llama:detokenize vocab tok-buf n-tokens
                                                text-buf n-needed
                                                remove-sp unparse-sp)))
              (cffi:foreign-string-to-lisp text-buf :count n-written))))))))


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
                                  top-k top-p min-p (seed 42)
                                  (parse-special t) prompt-tokens
                                  token-callback)
  "Generate text by continuing PROMPT. Returns two values: the generated string
and a stop reason (:eog, :length, or :callback).
Uses the context's model for tokenization. Blocks until EOS or MAX-TOKENS.
When PARSE-SPECIAL is true (default), special tokens in PROMPT are parsed
rather than treated as literal text — required for chat-template prompts.
When PROMPT-TOKENS is provided, it is used directly (skipping tokenization of
PROMPT). Use TOKENIZE-CHAT to build safe token sequences for chat prompts.
When TOKEN-CALLBACK is provided, it is called with each decoded token string
as it is produced. Return NIL from the callback to stop generation early."
  (with-fp-traps-masked
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
      (let ((sampler (build-sampler-chain :temp temp :top-k top-k :top-p top-p
                                          :min-p min-p :seed seed))
            (emitted-len 0))
        (unwind-protect
            (loop for i from 0 below max-tokens
                  for new-token = (%llama:sampler-sample sampler ctx -1)
                  do (%llama:sampler-accept sampler new-token)
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

(defun model-chat-template (model &optional name)
  "Return the chat template string embedded in MODEL.
If NAME is given, look up a specific named template."
  (with-fp-traps-masked
    (%llama:model-chat-template model (or name (cffi:null-pointer)))))

(defun list-chat-templates ()
  "Return a list of built-in chat template name strings."
  (let ((n (%llama:chat-builtin-templates (cffi:null-pointer) 0)))
    (when (> n 0)
      (cffi:with-foreign-object (output :pointer n)
        (%llama:chat-builtin-templates output n)
        (loop for i below n
              collect (cffi:foreign-string-to-lisp
                       (cffi:mem-aref output :pointer i)))))))

(defun format-chat (model messages &key template (add-assistant-prefix t))
  "Format MESSAGES as a chat prompt string using a Jinja-style chat template.
MESSAGES is a list of plists with :role and :content keys.
Uses MODEL's embedded chat template unless TEMPLATE is provided."
  (with-fp-traps-masked
    (let* ((tmpl (or template (model-chat-template model)))
           (tmpl-arg (or tmpl (cffi:null-pointer)))
           ;; Gemma templates use "model" instead of "assistant"
           (model-role (and (stringp tmpl) (search "'model'" tmpl)))
           (n-msg (length messages))
           (add-ass (if add-assistant-prefix 1 0))
           (msg-size (cffi:foreign-type-size '(:struct %llama:chat-message)))
           (foreign-strings nil))
      (cffi:with-foreign-object (chat '(:struct %llama:chat-message) n-msg)
        (unwind-protect
            (progn
              (loop for msg in messages
                    for i from 0
                    for msg-ptr = (cffi:inc-pointer chat (* i msg-size))
                    for role = (let ((r (getf msg :role)))
                                 (if (and model-role (string= r "assistant"))
                                     "model" r))
                    for role-ptr = (cffi:foreign-string-alloc role)
                    for content-ptr = (cffi:foreign-string-alloc (getf msg :content))
                    do (push role-ptr foreign-strings)
                       (push content-ptr foreign-strings)
                       (setf (cffi:mem-ref msg-ptr :pointer 0) role-ptr
                             (cffi:mem-ref msg-ptr :pointer 8) content-ptr))
              (let ((n-needed (%llama:chat-apply-template
                               tmpl-arg chat n-msg add-ass
                               (cffi:null-pointer) 0)))
                (when (< n-needed 0)
                  (error 'chat-template-error))
                (cffi:with-foreign-pointer-as-string (buf (1+ n-needed))
                  (%llama:chat-apply-template
                   tmpl-arg chat n-msg add-ass
                   buf (1+ n-needed)))))
          (dolist (ptr foreign-strings)
            (cffi:foreign-string-free ptr)))))))

(defun tokenize-chat (model messages &key template (add-assistant-prefix t))
  "Tokenize a chat conversation safely. Template markers are parsed as special
tokens; message content is not. This prevents content that resembles special
tokens (e.g. a model hallucinating <end_of_turn>) from corrupting the prompt
on subsequent turns. Returns a token vector suitable for GENERATE's :prompt-tokens."
  (let* ((formatted (format-chat model messages
                      :template template
                      :add-assistant-prefix add-assistant-prefix))
         (vocab (%llama:model-get-vocab model))
         (bos (%llama:token-bos vocab))
         (all-tokens (make-array 0 :element-type 'fixnum
                                   :adjustable t :fill-pointer 0))
         (pos 0))
    ;; Explicitly prepend BOS and skip the template's bos_token rendering
    (vector-push-extend bos all-tokens)
    (let ((bos-text (detokenize model (make-array 1 :element-type 'fixnum
                                                    :initial-element bos))))
      (when (and (plusp (length bos-text))
                 (eql 0 (search bos-text formatted)))
        (setf pos (length bos-text))))
    (dolist (msg messages)
      (let* ((content (getf msg :content))
             (content-start (search content formatted :start2 pos)))
        (when content-start
          (when (> content-start pos)
            (let ((toks (tokenize model (subseq formatted pos content-start)
                          :add-special nil :parse-special t)))
              (loop for tok across toks do (vector-push-extend tok all-tokens))))
          (when (plusp (length content))
            (let ((toks (tokenize model content
                          :add-special nil :parse-special nil)))
              (loop for tok across toks do (vector-push-extend tok all-tokens))))
          (setf pos (+ content-start (length content))))))
    (when (< pos (length formatted))
      (let ((toks (tokenize model (subseq formatted pos)
                    :add-special nil :parse-special t)))
        (loop for tok across toks do (vector-push-extend tok all-tokens))))
    (let ((result (make-array (length all-tokens) :element-type 'fixnum)))
      (dotimes (i (length all-tokens))
        (setf (aref result i) (aref all-tokens i)))
      result)))

;;; LoRA adapter wrappers

(defmacro with-lora ((var model path) &body body)
  "Load a LoRA adapter from PATH for MODEL, bind it to VAR, execute BODY, free the adapter."
  (let ((adapter-ptr (gensym "ADAPTER"))
        (path-val (gensym "PATH")))
    `(with-fp-traps-masked
       (let* ((,path-val ,path)
              (,adapter-ptr (%llama:adapter-lora-init ,model ,path-val)))
         (when (cffi:null-pointer-p ,adapter-ptr)
           (error 'lora-load-error :path ,path-val))
         (let ((,var ,adapter-ptr))
           (unwind-protect
                (progn ,@body)
             (%llama:adapter-lora-free ,var)))))))

(defun apply-lora (ctx adapter &key (scale 1.0))
  "Attach a LoRA ADAPTER to context CTX with the given SCALE factor.
Returns NIL on success, signals LORA-APPLY-ERROR on failure."
  (with-fp-traps-masked
    (let ((scale-f (coerce scale 'single-float)))
      (cffi:with-foreign-objects ((adapters-buf :pointer 1)
                                  (scales-buf :float 1))
        (setf (cffi:mem-aref adapters-buf :pointer 0) adapter)
        (setf (cffi:mem-aref scales-buf :float 0) scale-f)
        (let ((rc (%llama:set-adapters-lora ctx adapters-buf 1 scales-buf)))
          (unless (zerop rc)
            (error 'lora-apply-error :code rc))
          nil)))))

(defun read-adapter-meta-string (adapter index reader-fn)
  "Read a metadata string from ADAPTER at INDEX using READER-FN.
READER-FN is called as (funcall reader-fn adapter index buf buf-size)."
  (let ((buf-size 256))
    (cffi:with-foreign-pointer-as-string (buf buf-size)
      (let ((n (funcall reader-fn adapter index buf buf-size)))
        (when (> n buf-size)
          (setf buf-size n)
          (cffi:with-foreign-pointer-as-string (buf2 buf-size)
            (funcall reader-fn adapter index buf2 buf-size)
            (return-from read-adapter-meta-string
              (cffi:foreign-string-to-lisp buf2 :count (min n buf-size)))))
        (cffi:foreign-string-to-lisp buf :count (max 0 n))))))

(defun lora-metadata (adapter)
  "Return metadata from ADAPTER as an alist of (key . value) string pairs."
  (with-fp-traps-masked
    (let ((count (%llama:adapter-meta-count adapter)))
      (loop for i from 0 below count
            collect (cons
                     (read-adapter-meta-string
                      adapter i #'%llama:adapter-meta-key-by-index)
                     (read-adapter-meta-string
                      adapter i #'%llama:adapter-meta-val-str-by-index))))))
