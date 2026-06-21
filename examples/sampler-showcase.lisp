;;; Grammar / constrained generation showcase — demonstrates GBNF grammar-
;;; constrained sampling, lazy grammar activation with trigger words, the
;;; fill-in-the-middle sampler, and grammar-error condition handling.
;;;
;;; Each demo narrates what it does before running, so reading the output
;;; explains the API.
;;;
;;; Setup:
;;;   export LLAMA_MODEL=/path/to/model.gguf    ; or set *model-path* in the REPL
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/sampler-showcase.lisp")
;;;   (cl-llama-cpp/examples/sampler-showcase:run)

(defpackage #:cl-llama-cpp/examples/sampler-showcase
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/sampler-showcase)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

(defparameter *json-grammar*
  "root    ::= \"{\" ws members ws \"}\"
members ::= pair (\",\" ws pair)*
pair    ::= string ws \":\" ws value
value   ::= string | number | \"true\" | \"false\" | \"null\"
string  ::= \"\\\"\" [a-zA-Z0-9_ ]* \"\\\"\"
number  ::= [0-9]+
ws      ::= [ ]*")

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun stream-generate (ctx prompt &key sampler (max-tokens 128))
  "Call GENERATE with SAMPLER and a streaming token-callback, printing each
token to stdout as it arrives.  Returns the generated text."
  (multiple-value-bind (text stop-reason)
      (generate ctx prompt
                :max-tokens max-tokens
                :sampler sampler
                :token-callback (lambda (tok)
                                  (write-string tok)
                                  (force-output)
                                  t))
    (terpri)
    (values text stop-reason)))

;;; ── Demo 1: Strict grammar ─────────────────────────────────────────

(defun demo-strict-grammar (model ctx)
  (banner "DEMO 1: Strict Grammar — Force JSON Output")

  (format t "A GBNF grammar constrains every generated token to match the~%")
  (format t "grammar rules.  This guarantees structurally valid output.~2%")
  (format t "Grammar (GBNF):~%~A~2%" *json-grammar*)

  ;; 1a — The easy way: pass :grammar to generate
  (format t "── 1a: generate with :grammar (recommended) ──~2%")
  (format t "Prompt: \"Output a JSON object describing a cat.\"~2%")
  (multiple-value-bind (text stop-reason)
      (generate ctx "Output a JSON object describing a cat."
                :max-tokens 128 :temp 0.3
                :grammar *json-grammar*
                :token-callback (lambda (tok)
                                  (write-string tok)
                                  (force-output)
                                  t))
    (format t "~2%Stop reason: ~A~%" stop-reason)
    (format t "Result: ~S~2%" text))

  ;; 1b — Lower-level: make-grammar-sampler + manual chain
  (format t "── 1b: make-grammar-sampler + manual chain ──~2%")
  (format t "with-sampler-chain (no keyword args) allocates an empty chain.~%")
  (format t "sampler-chain-add accepts any typed LLAMA-SAMPLER handle.~%")
  (format t "make-temp-sampler / make-dist-sampler return typed handles.~%")
  (format t "The chain takes ownership — freeing the chain frees all its samplers.~%")
  (format t "generate :sampler borrows the chain without freeing it.~2%")
  (format t "Prompt: \"Output a JSON object describing a dog.\"~2%")
  (let ((gs (make-grammar-sampler model *json-grammar*)))
    (with-sampler-chain (chain)
      (sampler-chain-add chain gs)
      (sampler-chain-add chain (make-temp-sampler 0.3))
      (sampler-chain-add chain (make-dist-sampler 42))
      (stream-generate ctx "Output a JSON object describing a dog."
                       :sampler chain :max-tokens 128)))

  ;; 1c — with-grammar-sampler resource macro
  (format t "~%── 1c: with-grammar-sampler (resource macro) ──~2%")
  (format t "with-grammar-sampler creates a sampler and frees it on scope~%")
  (format t "exit — useful for inspection or single-sampler workflows.~2%")
  (with-grammar-sampler (gs model *json-grammar*)
    (format t "  Sampler handle:  ~A~%" gs)
    (format t "  Valid handle:    ~A~%" (llama-sampler-p gs)))
  (format t "  Sampler automatically freed on scope exit.~%"))

;;; ── Demo 2: Lazy grammar ───────────────────────────────────────────

(defun demo-lazy-grammar (model ctx)
  (banner "DEMO 2: Lazy Grammar — Natural to Structured Mid-Stream")

  (format t "A lazy grammar sampler stays inactive during free-form generation.~%")
  (format t "When a trigger word appears in the output (here \"{\"), the grammar~%")
  (format t "activates and constrains all subsequent tokens to match.~2%")
  (format t "This lets the model write a natural sentence and then seamlessly~%")
  (format t "transition into strictly valid JSON.~2%")

  (format t "Building chain: with-sampler-chain :grammar-lazy t~%")
  (format t "                                   :grammar-trigger-words '(\"{\")~2%")
  (format t "Prompt: \"Describe a dog in one sentence, then output traits as JSON:\"~2%")

  (with-sampler-chain (chain :model model
                             :grammar *json-grammar*
                             :grammar-lazy t
                             :grammar-trigger-words '("{")
                             :temp 0.3)
    (stream-generate ctx "Describe a dog in one sentence, then output traits as JSON:"
                     :sampler chain :max-tokens 256))
  (format t "~%The grammar activated when \"{\" appeared, ensuring valid JSON.~%"))

;;; ── Demo 3: Error handling ─────────────────────────────────────────

(defun demo-error-handling (model)
  (banner "DEMO 3: Error Handling — Catching grammar-error")

  (format t "make-grammar-sampler validates its input and signals grammar-error~%")
  (format t "when the grammar is empty or the C library rejects it.~2%")

  (format t "Test 1: Empty grammar string~%")
  (let ((caught nil))
    (handler-case
        (make-grammar-sampler model "")
      (grammar-error (c)
        (setf caught t)
        (format t "  Caught: ~A~%" c)
        (format t "  Type:   ~A~%" (type-of c))
        (format t "  Slot:   ~S~2%" (grammar-error-grammar c))))
    (assert caught () "Expected grammar-error for empty grammar string"))

  (format t "Test 2: Invalid GBNF syntax~%")
  (handler-case
      (let ((sampler (make-grammar-sampler model "%%%not-valid-gbnf%%%")))
        (format t "  C library accepted the string (returned non-null sampler).~%")
        (format t "  Freeing sampler with free-sampler.~2%")
        (free-sampler sampler))
    (grammar-error (c)
      (format t "  Caught: ~A~2%" c)))

  (format t "Grammar errors are caught cleanly — no crashes or leaks.~%"))

;;; ── Entry point ────────────────────────────────────────────────────

(defun run ()
  "Run all grammar sampler demos."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (format t "~&Loading model: ~A~%" *model-path*)
  (with-model (model *model-path* :n-gpu-layers 99)
    (with-context (ctx model :n-ctx 2048)
      (demo-strict-grammar model ctx)
      (demo-lazy-grammar model ctx)
      (demo-error-handling model)))
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  All demos complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
