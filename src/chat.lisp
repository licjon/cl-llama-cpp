(in-package #:cl-llama-cpp)

(llama-defun model-chat-template (model &optional name)
  "Return the chat template string embedded in MODEL.
If NAME is given, look up a specific named template."
  (let ((res-ptr (if name
                     (cffi:with-foreign-string (name-ptr name)
                       (%llama:model-chat-template (llama-model-pointer model) name-ptr))
                     (%llama:model-chat-template (llama-model-pointer model) (cffi:null-pointer)))))
    (unless (cffi:null-pointer-p res-ptr)
      (cffi:foreign-string-to-lisp res-ptr))))

(llama-defun list-chat-templates ()
  "Return a list of built-in chat template name strings."
  (let ((n (%llama:chat-builtin-templates (cffi:null-pointer) 0)))
    (when (> n 0)
      (cffi:with-foreign-object (output :pointer n)
        (%llama:chat-builtin-templates output n)
        (loop for i below n
              collect (cffi:foreign-string-to-lisp
                       (cffi:mem-aref output :pointer i)))))))

(llama-defun format-chat (model messages &key template (add-assistant-prefix t))
  "Format MESSAGES as a chat prompt string using a Jinja-style chat template.
MESSAGES is a list of plists with :role and :content keys.
Uses MODEL's embedded chat template unless TEMPLATE is provided.
Signals INPUT-VALIDATION-ERROR if MESSAGES is empty or malformed."
  (when (endp messages)
    (error 'input-validation-error
           :function-name 'format-chat :argument :messages :value messages
           :reason "messages must be a non-empty list"))
  (dolist (msg messages)
    (unless (and (getf msg :role) (stringp (getf msg :role)))
      (error 'input-validation-error
             :function-name 'format-chat :argument :messages :value msg
             :reason "each message must have a :ROLE string"))
    (unless (and (getf msg :content) (stringp (getf msg :content)))
      (error 'input-validation-error
             :function-name 'format-chat :argument :messages :value msg
             :reason "each message must have a :CONTENT string")))
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
              (if (< n-needed 0)
                  (restart-case (error 'chat-template-error)
                    (use-default-template ()
                      :report "Retry format-chat with the model's default template"
                      (format-chat model messages
                                   :add-assistant-prefix add-assistant-prefix)))
                  (cffi:with-foreign-pointer-as-string (buf (1+ n-needed))
                    (%llama:chat-apply-template
                     tmpl-arg chat n-msg add-ass
                     buf (1+ n-needed))))))
        (dolist (ptr foreign-strings)
          (cffi:foreign-string-free ptr))))))

(llama-defun tokenize-chat (model messages &key template (add-assistant-prefix t))
  "Tokenize a chat conversation safely. Template markers are parsed as special
tokens; message content is not. This prevents content that resembles special
tokens (e.g. a model hallucinating <end_of_turn>) from corrupting the prompt
on subsequent turns. Returns a token vector suitable for GENERATE's :prompt-tokens."
  (let* ((formatted (format-chat model messages
                      :template template
                      :add-assistant-prefix add-assistant-prefix))
         (vocab (%llama:model-get-vocab (llama-model-pointer model)))
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
