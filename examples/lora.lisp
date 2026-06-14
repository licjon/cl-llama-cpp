;;; LoRA adapter example — demonstrates with-lora, apply-lora, and lora-metadata.
;;;
;;; This walks through using the Naomarik/pirate-gemma3-1b adapter with
;;; gemma-3-1b-it.  The adapter shifts the model toward a more casual,
;;; conversational tone (not literal pirate dialect despite the name).
;;; Use COMPARE to see the difference side-by-side.
;;;
;;; Setup (run once):
;;;
;;;   1. Get a base model GGUF (gemma-3-1b-it shown here, any chat model works):
;;;
;;;        pip install huggingface-hub
;;;        huggingface-cli download google/gemma-3-1b-it-GGUF \
;;;          --include "gemma-3-1b-it-Q4_K_M.gguf" --local-dir ~/models
;;;
;;;   2. Download the LoRA adapter:
;;;
;;;        huggingface-cli download Naomarik/pirate-gemma3-1b \
;;;          --local-dir ~/models/pirate-gemma3-1b
;;;
;;;   3. Convert the adapter from safetensors to GGUF:
;;;
;;;        # The adapter dir also contains a merged model.safetensors which
;;;        # must be moved aside — the converter indexes all safetensors
;;;        # files and will silently produce a corrupt GGUF otherwise.
;;;        cd <path-to-cl-llama-cpp>/llama.cpp
;;;        pip install -r requirements/requirements-convert_lora_to_gguf.txt
;;;        mv ~/models/pirate-gemma3-1b/model.safetensors \
;;;           ~/models/pirate-gemma3-1b/model.safetensors.bak
;;;        python convert_lora_to_gguf.py \
;;;          --base ~/models/pirate-gemma3-1b \
;;;          --outfile ~/models/pirate-lora.gguf \
;;;          ~/models/pirate-gemma3-1b
;;;        mv ~/models/pirate-gemma3-1b/model.safetensors.bak \
;;;           ~/models/pirate-gemma3-1b/model.safetensors
;;;
;;;   4. Set environment variables and run:
;;;
;;;        export LLAMA_MODEL=~/models/gemma-3-1b-it-Q4_K_M.gguf
;;;        export LLAMA_LORA=~/models/pirate-lora.gguf
;;;
;;;        (ql:quickload :cl-llama-cpp)
;;;        (load "examples/lora.lisp")
;;;        (cl-llama-cpp/examples/lora:run)
;;;        (cl-llama-cpp/examples/lora:compare)

(defpackage #:cl-llama-cpp/examples/lora
  (:use #:cl #:cl-llama-cpp)
  (:export #:run #:compare))

(in-package #:cl-llama-cpp/examples/lora)

(defparameter *model-path*
  (or (uiop:getenv "LLAMA_MODEL")
      (error "Set LLAMA_MODEL to the path of a GGUF model.")))

(defparameter *lora-path*
  (or (uiop:getenv "LLAMA_LORA")
      (error "Set LLAMA_LORA to the path of a LoRA adapter GGUF.")))

(defun run (&optional (prompt "Tell me about the Common Lisp programming language."))
  "Chat with the LoRA-adapted model, streaming to stdout."
  (with-model (model *model-path* :n-gpu-layers 99)
    (with-context (ctx model :n-ctx 4096)
      (with-lora (adapter model *lora-path*)
        (format t "~&LoRA metadata:~%")
        (dolist (entry (lora-metadata adapter))
          (format t "  ~A: ~A~%" (car entry) (cdr entry)))
        (terpri)
        (apply-lora ctx adapter)
        (let* ((messages (list (list :role "user" :content prompt)))
               (tokens (tokenize-chat model messages)))
          (generate ctx nil
            :prompt-tokens tokens
            :max-tokens 512
            :temp 0.7
            :token-callback (lambda (tok)
                              (write-string tok)
                              (force-output)
                              t))
          (terpri)
          (values))))))

(defun compare (&optional (prompt "Tell me about the Common Lisp programming language."))
  "Run the same prompt with and without LoRA to see the difference."
  (with-model (model *model-path* :n-gpu-layers 99)
    (let* ((messages (list (list :role "user" :content prompt)))
           (tokens (tokenize-chat model messages)))

      (format t "~%=== Base model ===~%~%")
      (with-context (ctx model :n-ctx 4096)
        (generate ctx nil
          :prompt-tokens tokens
          :max-tokens 512
          :temp 0.7
          :token-callback (lambda (tok)
                            (write-string tok)
                            (force-output)
                            t)))
      (terpri)

      (format t "~%=== With LoRA ===~%~%")
      (with-context (ctx model :n-ctx 4096)
        (with-lora (adapter model *lora-path*)
          (apply-lora ctx adapter)
          (generate ctx nil
            :prompt-tokens tokens
            :max-tokens 512
            :temp 0.7
            :token-callback (lambda (tok)
                              (write-string tok)
                              (force-output)
                              t))))
      (terpri)
      (values))))
