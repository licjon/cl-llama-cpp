(in-package #:cl-llama-cpp/common/chat)

;;; Constants (match C++ enum values)

(defconstant +chat-format-content-only+ 0)
(defconstant +chat-format-peg-simple+   1)
(defconstant +chat-format-peg-native+   2)
(defconstant +chat-format-peg-gemma4+   3)

(defconstant +tool-choice-auto+     0)
(defconstant +tool-choice-required+ 1)
(defconstant +tool-choice-none+     2)

(defconstant +reasoning-format-none+            0)
(defconstant +reasoning-format-auto+            1)
(defconstant +reasoning-format-deepseek-legacy+ 2)
(defconstant +reasoning-format-deepseek+        3)

(defconstant +continuation-none+      0)
(defconstant +continuation-auto+      1)
(defconstant +continuation-reasoning+ 2)
(defconstant +continuation-content+   3)

;;; Conditions

(define-condition chat-init-error (cl-llama-cpp:llama-error)
  ((message :initarg :message :reader chat-init-error-message))
  (:report (lambda (c s)
             (format s "Chat templates init failed: ~A"
                     (chat-init-error-message c)))))

(define-condition chat-parse-error (cl-llama-cpp:llama-error)
  ((message :initarg :message :reader chat-parse-error-message))
  (:report (lambda (c s)
             (format s "Chat parse failed: ~A"
                     (chat-parse-error-message c)))))

;;; Handle struct

(defstruct (chat-templates
             (:constructor %make-chat-templates)
             (:copier nil))
  (pointer (cffi:null-pointer) :type cffi:foreign-pointer)
  (freed-cell (list nil) :type cons :read-only t))

(defun %try-claim-for-free (cell)
  #+sbcl (null (sb-ext:cas (car cell) nil t))
  #-sbcl (prog1 (null (car cell))
           (setf (car cell) t)))

;;; JSON marshalling helpers

(defun %msg-to-ht (msg)
  (let ((ht (make-hash-table :test 'equal)))
    (setf (gethash "role" ht) (getf msg :role))
    (let ((content (getf msg :content)))
      (when content
        (setf (gethash "content" ht) content)))
    (let ((tool-calls (getf msg :tool-calls)))
      (when tool-calls
        (setf (gethash "tool_calls" ht)
              (mapcar (lambda (tc)
                        (let ((tc-ht (make-hash-table :test 'equal))
                              (fn-ht (make-hash-table :test 'equal)))
                          (setf (gethash "name" fn-ht) (getf tc :name))
                          (setf (gethash "arguments" fn-ht) (getf tc :arguments))
                          (setf (gethash "type" tc-ht) "function")
                          (setf (gethash "function" tc-ht) fn-ht)
                          (let ((id (getf tc :id)))
                            (when id (setf (gethash "id" tc-ht) id)))
                          tc-ht))
                      tool-calls))))
    (let ((tool-call-id (getf msg :tool-call-id)))
      (when tool-call-id
        (setf (gethash "tool_call_id" ht) tool-call-id)))
    (let ((reasoning (getf msg :reasoning-content)))
      (when reasoning
        (setf (gethash "reasoning_content" ht) reasoning)))
    ht))

(defun %msgs-to-json (messages)
  (if (null messages)
      "[]"
      (with-output-to-string (s)
        (yason:encode (mapcar #'%msg-to-ht messages) s))))

(defun %tool-to-ht (tool)
  (let ((ht (make-hash-table :test 'equal))
        (fn-ht (make-hash-table :test 'equal)))
    (setf (gethash "name" fn-ht) (getf tool :name))
    (setf (gethash "description" fn-ht) (or (getf tool :description) ""))
    (let ((params (getf tool :parameters)))
      (when params
        (setf (gethash "parameters" fn-ht)
              (etypecase params
                (string (yason:parse params))
                (hash-table params)))))
    (setf (gethash "type" ht) "function")
    (setf (gethash "function" ht) fn-ht)
    ht))

(defun %tools-to-json (tools)
  (if (null tools)
      "[]"
      (with-output-to-string (s)
        (yason:encode (mapcar #'%tool-to-ht tools) s))))

(defun %json-to-msg-plist (ht)
  (let* ((content (gethash "content" ht))
         (result (list :role (or (gethash "role" ht) "assistant")
                       :content (if (or (null content) (eq content :null))
                                    ""
                                    content))))
    (let ((tool-calls (gethash "tool_calls" ht)))
      (when (and tool-calls (not (eq tool-calls :null)))
        (setf (getf result :tool-calls)
              (map 'list
                   (lambda (tc)
                     (let ((fn (or (gethash "function" tc) tc)))
                       (append
                        (list :name (gethash "name" fn)
                              :arguments (gethash "arguments" fn))
                        (let ((id (or (gethash "id" tc) (gethash "id" fn))))
                          (when (and id (not (eq id :null)))
                            (list :id id))))))
                   tool-calls))))
    (let ((reasoning (gethash "reasoning_content" ht)))
      (when (and reasoning (not (eq reasoning :null)) (not (equal reasoning "")))
        (setf (getf result :reasoning-content) reasoning)))
    (let ((tool-call-id (gethash "tool_call_id" ht)))
      (when (and tool-call-id (not (eq tool-call-id :null)))
        (setf (getf result :tool-call-id) tool-call-id)))
    result))

;;; Result processing

(defun %process-chat-result (result error-type)
  (let ((output-ptr (getf result 'output))
        (status (getf result 'status)))
    (unwind-protect
         (progn
           (when (cffi:null-pointer-p output-ptr)
             (error error-type
                    :message "shim returned NULL (out of memory?)"))
           (let ((output-string (cffi:foreign-string-to-lisp output-ptr)))
             (if (zerop status)
                 output-string
                 (error error-type :message output-string))))
      (unless (cffi:null-pointer-p output-ptr)
        (%shim-free output-ptr)))))

;;; Lifecycle

(defun make-chat-templates (model &key template-override
                                       (bos-override "")
                                       (eos-override ""))
  "Create a chat templates handle from a cl-llama-cpp model.
The handle MUST NOT outlive the model it was created from."
  (let ((result (%chat-init (cl-llama-cpp:llama-model-pointer model)
                            (or template-override "")
                            bos-override eos-override)))
    (let ((handle-ptr (getf result 'handle))
          (err-ptr (getf result 'error)))
      (unwind-protect
           (cond
             ((and (not (cffi:null-pointer-p handle-ptr))
                   (cffi:null-pointer-p err-ptr))
              (let ((templates (%make-chat-templates :pointer handle-ptr)))
                (%register-finalizer templates)
                templates))
             ((not (cffi:null-pointer-p err-ptr))
              (error 'chat-init-error
                     :message (cffi:foreign-string-to-lisp err-ptr)))
             (t
              (error 'chat-init-error
                     :message "init returned NULL without error message")))
        (unless (cffi:null-pointer-p err-ptr)
          (%shim-free err-ptr))))))

(defun free-chat-templates (templates)
  "Free a chat templates handle. Idempotent."
  (when (%try-claim-for-free (chat-templates-freed-cell templates))
    (tg:cancel-finalization templates)
    (%chat-free (chat-templates-pointer templates)))
  nil)

(defun %register-finalizer (templates)
  (let ((ptr (chat-templates-pointer templates))
        (cell (chat-templates-freed-cell templates)))
    (tg:finalize templates
      (lambda ()
        (when (%try-claim-for-free cell)
          (ignore-errors (%chat-free ptr)))))))

(defmacro with-chat-templates ((var model &rest keys
                                    &key template-override bos-override
                                         eos-override)
                                &body body)
  "Create a chat templates handle, bind to VAR, execute BODY, free."
  (declare (ignore template-override bos-override eos-override))
  `(let ((,var (make-chat-templates ,model ,@keys)))
     (unwind-protect (progn ,@body)
       (free-chat-templates ,var))))

;;; Queries

(defun chat-templates-source (templates &key (variant ""))
  "Return the chat template source string."
  (let ((ptr (%chat-templates-source (chat-templates-pointer templates) variant)))
    (if (cffi:null-pointer-p ptr)
        (error 'chat-init-error :message "failed to retrieve template source")
        (unwind-protect
             (cffi:foreign-string-to-lisp ptr)
          (%shim-free ptr)))))

;;; Core API

(defun chat-templates-apply (templates messages &key
                              (tools nil)
                              (tool-choice +tool-choice-auto+)
                              (add-generation-prompt t)
                              (use-jinja t)
                              (parallel-tool-calls nil)
                              (reasoning-format +reasoning-format-none+)
                              (enable-thinking t)
                              (json-schema "")
                              (grammar "")
                              (continue-final-message +continuation-none+))
  "Apply chat templates to MESSAGES, returning a plist with:
  :PROMPT — the rendered prompt string
  :GRAMMAR — GBNF grammar string for constrained generation
  :FORMAT — chat format enum value
  :GRAMMAR-LAZY — whether grammar is lazy
  :GENERATION-PROMPT — the generation prompt prefix
  :ADDITIONAL-STOPS — list of additional stop strings
  :GRAMMAR-TRIGGERS — list of grammar trigger plists
  :PRESERVED-TOKENS — list of preserved token strings
  :SUPPORTS-THINKING — whether thinking is supported
  :THINKING-START-TAG, :THINKING-END-TAG — tag strings

MESSAGES is a list of plists (:role :content &key :tool-calls :tool-call-id
:reasoning-content). TOOLS is a list of plists (:name :description :parameters)."
  (let* ((msgs-json (%msgs-to-json messages))
         (tools-json (if tools (%tools-to-json tools) ""))
         (result (%chat-apply (chat-templates-pointer templates)
                              msgs-json tools-json
                              tool-choice
                              (if add-generation-prompt 1 0)
                              (if use-jinja 1 0)
                              (if parallel-tool-calls 1 0)
                              reasoning-format
                              (if enable-thinking 1 0)
                              (or json-schema "") (or grammar "")
                              continue-final-message))
         (json-str (%process-chat-result result 'chat-init-error))
         (ht (yason:parse json-str)))
    (list :prompt (gethash "prompt" ht)
          :grammar (gethash "grammar" ht)
          :format (gethash "format" ht)
          :grammar-lazy (gethash "grammar_lazy" ht)
          :generation-prompt (gethash "generation_prompt" ht)
          :additional-stops (coerce (or (gethash "additional_stops" ht) #()) 'list)
          :grammar-triggers (map 'list
                                 (lambda (tr)
                                   (list :type (gethash "type" tr)
                                         :value (gethash "value" tr)
                                         :token (gethash "token" tr)))
                                 (or (gethash "grammar_triggers" ht) #()))
          :preserved-tokens (coerce (or (gethash "preserved_tokens" ht) #()) 'list)
          :supports-thinking (gethash "supports_thinking" ht)
          :thinking-start-tag (gethash "thinking_start_tag" ht)
          :thinking-end-tag (gethash "thinking_end_tag" ht))))

(defun %keyword-to-format (kw)
  (ecase kw
    (:content-only +chat-format-content-only+)
    (:peg-simple   +chat-format-peg-simple+)
    (:peg-native   +chat-format-peg-native+)
    (:peg-gemma4   +chat-format-peg-gemma4+)))

(defun chat-parse (input &key (is-partial nil)
                              (format :content-only)
                              (reasoning-format +reasoning-format-none+)
                              (parse-tool-calls t)
                              (generation-prompt "")
                              templates)
  "Parse model output text into a structured message plist.
Returns a plist with :ROLE, :CONTENT, and optionally :TOOL-CALLS,
:REASONING-CONTENT.

FORMAT is :CONTENT-ONLY, :PEG-SIMPLE, :PEG-NATIVE, or :PEG-GEMMA4.
If TEMPLATES is provided (a chat-templates handle after an apply call),
uses the cached parser params for PEG format support."
  (let* ((format-int (etypecase format
                       (keyword (%keyword-to-format format))
                       (integer format)))
         (result (if templates
                     (%chat-parse-cached (chat-templates-pointer templates)
                                         input
                                         (if is-partial 1 0)
                                         reasoning-format)
                     (%chat-parse-simple input
                                         (if is-partial 1 0)
                                         format-int
                                         reasoning-format
                                         (if parse-tool-calls 1 0)
                                         (or generation-prompt ""))))
         (json-str (%process-chat-result result 'chat-parse-error))
         (ht (yason:parse json-str)))
    (%json-to-msg-plist ht)))

(defun chat-format-single (templates new-message &key
                            (past-messages nil)
                            (add-assistant t)
                            (use-jinja t))
  "Format a single message, taking into account past messages.
Returns a formatted string.
TEMPLATES is a chat-templates handle.
NEW-MESSAGE and elements of PAST-MESSAGES are plists with :ROLE and :CONTENT."
  (let* ((past-json (if past-messages (%msgs-to-json past-messages) ""))
         (new-json (with-output-to-string (s)
                     (yason:encode (%msg-to-ht new-message) s)))
         (result (%chat-format-single (chat-templates-pointer templates)
                                      past-json new-json
                                      (if add-assistant 1 0)
                                      (if use-jinja 1 0))))
    (%process-chat-result result 'chat-init-error)))

(defun chat-verify-template (template &key (use-jinja t))
  "Check if a chat template string is valid.
Returns T if valid, NIL otherwise."
  (not (zerop (%chat-verify-template template (if use-jinja 1 0)))))
