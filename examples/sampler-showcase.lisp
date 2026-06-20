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

(defun sample-loop (ctx sampler model prompt &key (max-tokens 128))
  "Decode PROMPT into CTX, then sample up to MAX-TOKENS using SAMPLER,
streaming each token to stdout.  Returns the generated text."
  (with-llama-compatible-fp-environment
    (let* ((vocab (%llama:model-get-vocab model))
           (tokens (tokenize model prompt :parse-special t))
           (n (length tokens))
           (generated (make-array 0 :element-type 'fixnum
                                    :adjustable t :fill-pointer 0))
           (emitted-len 0))
      (%llama:memory-clear (%llama:get-memory ctx) 1)
      (cffi:with-foreign-object (buf '%llama:token n)
        (dotimes (i n)
          (setf (cffi:mem-aref buf '%llama:token i) (aref tokens i)))
        (%llama:decode ctx (%llama:batch-get-one buf n)))
      ;; sampler-sample calls sampler-accept internally — do NOT accept again
      (loop for i from 0 below max-tokens
            for tok = (%llama:sampler-sample sampler ctx -1)
            until (not (zerop (%llama:token-is-eog vocab tok)))
            do (vector-push-extend tok generated)
               (ignore-errors
                 (let* ((full (detokenize model generated :remove-special t))
                        (new-text (subseq full emitted-len)))
                   (when (plusp (length new-text))
                     (write-string new-text)
                     (force-output)
                     (setf emitted-len (length full)))))
               (cffi:with-foreign-object (buf '%llama:token 1)
                 (setf (cffi:mem-aref buf '%llama:token 0) tok)
                 (%llama:decode ctx (%llama:batch-get-one buf 1))))
      (terpri)
      (if (zerop (length generated))
          ""
          (detokenize model generated :remove-special t)))))

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
  (format t "For custom sampler chains, create the grammar sampler with~%")
  (format t "make-grammar-sampler and add it to a chain yourself.  The chain~%")
  (format t "takes ownership — freeing the chain frees all its samplers.~2%")
  (format t "Prompt: \"Output a JSON object describing a dog.\"~2%")
  (with-llama-compatible-fp-environment
    (let* ((gs (make-grammar-sampler model *json-grammar*))
           (chain (%llama:sampler-chain-init
                   (%llama:sampler-chain-default-params))))
      (%llama:sampler-chain-add chain gs)
      (%llama:sampler-chain-add chain (%llama:sampler-init-temp 0.3))
      (%llama:sampler-chain-add chain (%llama:sampler-init-dist 42))
      (unwind-protect
          (sample-loop ctx chain model
                       "Output a JSON object describing a dog."
                       :max-tokens 128)
        (%llama:sampler-free chain))))

  ;; 1c — with-grammar-sampler resource macro
  (format t "~%── 1c: with-grammar-sampler (resource macro) ──~2%")
  (format t "with-grammar-sampler creates a sampler and frees it on scope~%")
  (format t "exit — useful for inspection or single-sampler workflows.~2%")
  (with-grammar-sampler (gs model *json-grammar*)
    (format t "  Sampler pointer: ~A~%" gs)
    (format t "  Non-null:        ~A~%" (not (cffi:null-pointer-p gs))))
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
    (sample-loop ctx chain model
                 "Describe a dog in one sentence, then output traits as JSON:"
                 :max-tokens 256))
  (format t "~%The grammar activated when \"{\" appeared, ensuring valid JSON.~%"))

;;; ── Demo 3: Error handling ─────────────────────────────────────────

(defun demo-error-handling (model)
  (banner "DEMO 3: Error Handling — Catching grammar-error")

  (format t "make-grammar-sampler validates its input and signals grammar-error~%")
  (format t "when the grammar is empty or the C library rejects it.~2%")

  (format t "Test 1: Empty grammar string~%")
  (handler-case
      (make-grammar-sampler model "")
    (grammar-error (c)
      (format t "  Caught: ~A~%" c)
      (format t "  Type:   ~A~%" (type-of c))
      (format t "  Slot:   ~S~2%" (grammar-error-grammar c))))

  (format t "Test 2: Invalid GBNF syntax~%")
  (handler-case
      (let ((sampler (make-grammar-sampler model "%%%not-valid-gbnf%%%")))
        (format t "  C library accepted the string (returned non-null sampler).~%")
        (format t "  Freeing sampler.~2%")
        (with-llama-compatible-fp-environment (%llama:sampler-free sampler)))
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
