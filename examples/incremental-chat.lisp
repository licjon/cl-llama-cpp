;;;; incremental-chat.lisp
;;;;
;;;; An interactive chat example that reuses the KV cache across turns so each
;;;; turn decodes only the *new* tokens rather than the entire conversation
;;;; history.  Compare with examples/simple-chat.lisp, which re-prefills from
;;;; scratch every turn (O(n²) cost over a session).
;;;;
;;;; How this is fast:
;;;;
;;;;   MAKE-CHAT-SESSION establishes a stateful session that tracks exactly
;;;;   which tokens are resident in the context's KV cache.  CHAT-SESSION-SEND
;;;;   calls TOKENIZE-CHAT on the full message list (to pick up any template
;;;;   changes), finds the longest common prefix already in cache, evicts only
;;;;   the stale suffix if the prefix has changed, and calls GENERATE with
;;;;   :RESET-CONTEXT NIL so the decode continues from the current cache
;;;;   position.  Per-turn prefill cost is therefore proportional to the number
;;;;   of *new* tokens added each turn — not the total conversation length.
;;;;
;;;; Run from the REPL:
;;;;   (ql:quickload :cl-llama-cpp/examples)
;;;;   (cl-llama-cpp/examples/incremental-chat:main
;;;;     :model-path "/path/to/model.gguf"
;;;;     :n-gpu-layers 99)
;;;;
;;;; Or from a shell:
;;;;   ros -e '(ql:quickload :cl-llama-cpp/examples)' \
;;;;       -e '(cl-llama-cpp/examples/incremental-chat:main
;;;;             :model-path "/path/to/model.gguf")' -q
;;;;
;;;; Type "exit" or "quit" to end the session.

(defpackage #:cl-llama-cpp/examples/incremental-chat
  (:use #:cl #:cl-llama-cpp)
  (:export #:main))

(in-package #:cl-llama-cpp/examples/incremental-chat)

(defparameter *default-model-path*
  "/path/to/model.gguf")

(defun main (&key (model-path *default-model-path*)
                  (n-gpu-layers 99)
                  (n-ctx 4096)
                  (max-tokens 2048)
                  system-prompt)
  "Run an interactive incremental chat session using KV-cache reuse.
Loads MODEL-PATH, creates a CHAT-SESSION, then loops reading from standard
input until the user types 'exit' or 'quit'.

Each turn decodes only the new tokens — prefill latency stays constant
instead of growing with conversation length."
  (format t "Initializing llama.cpp backend...~%")
  (finish-output)

  (with-backend ()
    (set-log-callback (lambda (level text)
                        (when (>= level 3)
                          (format *error-output* "~a" text))))
    (with-model (model model-path :n-gpu-layers n-gpu-layers)
      (with-context (ctx model :n-ctx n-ctx)
        (let ((session (make-chat-session ctx :system-prompt system-prompt)))
          (format t "~%--- Incremental Chat Session Started (KV-cache reuse) ---~%")
          (format t "Type 'exit' or 'quit' to end.~%~%")
          (loop
            (format *query-io* "User> ")
            (finish-output *query-io*)
            (let ((input (read-line *query-io*)))
              (when (member input '("exit" "quit") :test #'string-equal)
                (return))
              (unless (uiop:emptyp (string-trim " " input))
                (format t "AI> ")
                (finish-output)
                (handler-case
                    (let ((reply (chat-session-send session input
                                                    :max-tokens max-tokens)))
                      (format t "~a~%~%" reply)
                      (finish-output))
                  (decode-error ()
                    (format t "~%[Context window full — resetting conversation. Please continue.]~%~%")
                    (finish-output)
                    (chat-session-reset session :keep-system t))))))
          (format t "~%--- Session Ended ---~%"))))))
