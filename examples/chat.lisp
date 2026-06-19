(defpackage #:cl-llama-cpp/examples/chat
  (:use #:cl #:cl-llama-cpp)
  (:export #:start-chat #:end-chat #:say #:clear-history))

(in-package #:cl-llama-cpp/examples/chat)

(defparameter *default-model-path*
  (or (uiop:getenv "LLAMA_MODEL")
      (error "Set LLAMA_MODEL to the path of a GGUF chat model.")))

(defvar *model* nil)
(defvar *ctx* nil)
(defvar *messages* nil)
(defvar *max-tokens* 2048)

(defun start-chat (&key (model-path *default-model-path*)
                        (n-gpu-layers 99) (n-ctx 4096) (max-tokens 2048))
  "Load a model and start a chat session. Use SAY to send messages, END-CHAT to clean up."
  (when *model* (end-chat))
  (cl-llama-cpp::ensure-backend)
  (with-llama-compatible-fp-environment
    (let* ((defaults (%llama:model-default-params))
           (params (cl-llama-cpp::override-params defaults
                     (list :n-gpu-layers n-gpu-layers)))
           (model (%llama:model-load-from-file model-path params)))
      (when (cffi:null-pointer-p model)
        (error 'model-load-error :path model-path))
      (let* ((ctx-defaults (%llama:context-default-params))
             (ctx-params (cl-llama-cpp::override-params ctx-defaults
                           (list :n-ctx n-ctx)))
             (ctx (%llama:new-context-with-model model ctx-params)))
        (when (cffi:null-pointer-p ctx)
          (%llama:model-free model)
          (error 'context-creation-error))
        (setf *model* model
              *ctx* ctx
              *messages* '()
              *max-tokens* max-tokens))))
  (values))

(defun end-chat ()
  "Free model and context resources."
  (when *ctx*
    (with-llama-compatible-fp-environment (%llama:free *ctx*))
    (setf *ctx* nil))
  (when *model*
    (with-llama-compatible-fp-environment (%llama:model-free *model*))
    (setf *model* nil))
  (setf *messages* nil)
  (values))

(defun clear-history ()
  "Clear conversation history without unloading the model."
  (setf *messages* nil)
  (values))

(defun trim-truncated (text)
  "Strip trailing incomplete formatting to avoid corrupting subsequent turns.
Strips trailing incomplete XML-like tags and the last partial line."
  (let ((s (string-right-trim '(#\Space #\Tab #\Newline #\Return) text)))
    ;; Strip trailing incomplete tag (< without matching >)
    (let ((open (position #\< s :from-end t))
          (close (position #\> s :from-end t)))
      (when (and open (or (not close) (> open close)))
        (setf s (string-right-trim '(#\Space #\Tab #\Newline #\Return)
                                   (subseq s 0 open)))))
    ;; Strip last partial line
    (let ((nl (position #\Newline s :from-end t)))
      (if nl (subseq s 0 nl) s))))

(defun say (message &key (max-tokens *max-tokens*))
  "Send MESSAGE, stream the response to *standard-output*."
  (unless *ctx* (error "No chat session — call START-CHAT first."))
  (setf *messages* (nconc *messages*
                          (list (list :role "user" :content message))))
  (let* ((prompt-tokens (tokenize-chat *model* *messages*))
         (reply (multiple-value-bind (text stop-reason)
                    (generate *ctx* nil
                      :prompt-tokens prompt-tokens
                      :max-tokens max-tokens
                      :token-callback (lambda (tok)
                                        (write-string tok)
                                        (force-output)
                                        t))
                  (if (eq stop-reason :length)
                      (trim-truncated text)
                      text))))
    (terpri)
    (setf *messages* (nconc *messages*
                            (list (list :role "assistant" :content reply))))
    (values)))
