;;;; simple-chat.lisp
;;;;
;;;; A minimal interactive chat example for cl-llama-cpp. MAIN loads a model,
;;;; then loops reading a line from standard input and printing the model's
;;;; reply, keeping the conversation history so the model has context. Each user
;;;; turn is rendered through the model's chat template (TOKENIZE-CHAT applies it
;;;; internally). Run it from a shell (e.g. `ros run ... main`) or call (main)
;;;; at the REPL; type "exit" or "quit" to end the session.
;;;;
;;;; Performance: this is the simple, not the fast, approach. Every turn the
;;;; ENTIRE conversation is re-tokenized and re-processed from scratch --
;;;; GENERATE clears the KV cache on each call and decodes the whole prompt
;;;; again. Cost therefore grows with the full history each turn (roughly O(n^2)
;;;; over a conversation), and per-turn latency climbs as the chat gets longer.
;;;; That is fine for a short demo, but for long sessions you would want
;;;; incremental decoding that reuses the KV cache and only processes the new
;;;; tokens each turn.

(defpackage #:cl-llama-cpp/examples/simple-chat
  (:use #:cl #:cl-llama-cpp)
  (:export #:main))

(in-package #:cl-llama-cpp/examples/simple-chat)

(defparameter *default-model-path*
  "/path/to/model.gguf")

(defun main (&key (model-path *default-model-path*)
                  (n-gpu-layers 99)
                  (n-ctx 4096)
                  (max-tokens 2048))
  "Run an interactive chat session. Loads MODEL-PATH, then loops reading a
line from standard input and printing the model's reply until the user
types 'exit' or 'quit'. Can be run from a shell (ros run ... main) or
called directly at the REPL with (main)."
  (format t "Initializing llama.cpp backend...~%")
  (finish-output)

  (with-backend ()
    ;; Silence llama.cpp's debug/info chatter (e.g. "CUDA Graph id N reused").
    ;; Only forward warnings and errors to stderr. llama_log_set also routes
    ;; ggml/CUDA backend messages, so this covers them too.
    (set-log-callback (lambda (level text)
                        (when (>= level 3)
                          (format *error-output* "~a" text))))
    (with-model (model model-path :n-gpu-layers n-gpu-layers)
      (with-context (ctx model :n-ctx n-ctx)
        (format t "~%--- Llama.cpp Chat Session Started ---~%")
        (format t "Type 'exit' or 'quit' to end.~%~%")
        (let ((messages '()))
          (loop
            (format t "User> ")
            (finish-output)
            (let ((input (read-line)))
              (when (member input '("exit" "quit") :test #'string-equal)
                (return))

              (unless (uiop:emptyp (string-trim " " input))
                (format t "AI> ")
                (finish-output)

                ;; Append the user turn, re-tokenize the whole conversation
                ;; through the model's chat template (tokenize-chat applies it
                ;; internally), generate, then record the reply.
                (setf messages
                      (nconc messages (list (list :role "user" :content input))))
                (let ((reply (generate ctx nil
                                       :prompt-tokens (tokenize-chat model messages)
                                       :max-tokens max-tokens)))
                  (format t "~a~%~%" reply)
                  (finish-output)
                  (setf messages
                        (nconc messages
                               (list (list :role "assistant" :content reply)))))))))
        (format t "~%--- Session Ended ---~%")))))
