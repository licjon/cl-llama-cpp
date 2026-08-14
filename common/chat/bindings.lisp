(in-package #:cl-llama-cpp/common/chat)

;;; Init / Free

(cffi:defcstruct chat-init-result
  (handle :pointer)
  (error :pointer))

(cffi:defcfun ("llama_extras_chat_init" %chat-init) (:struct chat-init-result)
  (model :pointer)
  (template-override :string)
  (bos-override :string)
  (eos-override :string))

(cffi:defcfun ("llama_extras_chat_free" %chat-free) :void
  (handle :pointer))

;;; Source

(cffi:defcfun ("llama_extras_chat_templates_source" %chat-templates-source) :pointer
  (handle :pointer)
  (variant :string))

;;; Apply / Parse / Format (all return chat-result)

(cffi:defcstruct chat-result
  (output :pointer)
  (status :int))

(cffi:defcfun ("llama_extras_chat_apply" %chat-apply) (:struct chat-result)
  (handle :pointer)
  (messages-json :string)
  (tools-json :string)
  (tool-choice :int)
  (add-generation-prompt :int)
  (use-jinja :int)
  (parallel-tool-calls :int)
  (reasoning-format :int)
  (enable-thinking :int)
  (json-schema :string)
  (grammar :string)
  (continue-final-message :int))

(cffi:defcfun ("llama_extras_chat_parse_cached" %chat-parse-cached) (:struct chat-result)
  (handle :pointer)
  (input :string)
  (is-partial :int)
  (reasoning-format :int))

(cffi:defcfun ("llama_extras_chat_parse_simple" %chat-parse-simple) (:struct chat-result)
  (input :string)
  (is-partial :int)
  (format :int)
  (reasoning-format :int)
  (parse-tool-calls :int)
  (generation-prompt :string))

(cffi:defcfun ("llama_extras_chat_format_single" %chat-format-single) (:struct chat-result)
  (handle :pointer)
  (past-messages-json :string)
  (new-message-json :string)
  (add-assistant :int)
  (use-jinja :int))

;;; Verify template

(cffi:defcfun ("llama_extras_chat_verify_template" %chat-verify-template) :int
  (tmpl :string)
  (use-jinja :int))

;;; Free

(cffi:defcfun ("llama_extras_chat_shim_free" %shim-free) :void
  (ptr :pointer))
