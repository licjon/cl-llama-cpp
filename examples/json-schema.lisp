;;;; json-schema.lisp
;;;;
;;;; Demonstrates JSON Schema constrained generation using
;;;; cl-llama-cpp/common/json-schema. Instead of writing GBNF grammars by hand,
;;;; you pass a standard JSON Schema string and the library converts it to GBNF
;;;; automatically via llama.cpp's json_schema_to_grammar.
;;;;
;;;; Three demos:
;;;;   1. Basic conversion — inspect the generated GBNF from a JSON Schema
;;;;   2. Constrained generation — generate structured JSON matching a schema
;;;;   3. Error handling — conditions, restarts, and recovery
;;;;
;;;; Setup:
;;;;   (ql:quickload :cl-llama-cpp/common/examples)
;;;;   (setf cl-llama-cpp/common/examples/json-schema::*model-path*
;;;;         "/path/to/model.gguf")
;;;;   (cl-llama-cpp/common/examples/json-schema:run)
;;;;
;;;; Or via environment variable:
;;;;   export LLAMA_MODEL=/path/to/model.gguf

(defpackage #:cl-llama-cpp/common/examples/json-schema
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/common/examples/json-schema)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

;;; ── Schemas ─────────────────────────────────────────────────────────

(defparameter *boolean-schema* "{\"type\":\"boolean\"}")

(defparameter *person-schema*
  "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"},\"hobbies\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"name\",\"age\",\"hobbies\"]}")

(defparameter *sentiment-schema*
  "{\"type\":\"object\",\"properties\":{\"sentiment\":{\"type\":\"string\",\"enum\":[\"positive\",\"negative\",\"neutral\"]},\"confidence\":{\"type\":\"number\"}},\"required\":[\"sentiment\",\"confidence\"]}")

;;; ── Helpers ─────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

;;; ── Demo 1: Schema → GBNF conversion ───────────────────────────────

(defun demo-conversion ()
  (banner "DEMO 1: JSON Schema → GBNF Conversion")

  (format t "json-schema-to-grammar converts a JSON Schema string into a GBNF~%")
  (format t "grammar string that llama.cpp can use for constrained sampling.~2%")

  (format t "── Boolean schema ──~%")
  (format t "Input:  ~A~%" *boolean-schema*)
  (let ((gbnf (cl-llama-cpp/common/json-schema:json-schema-to-grammar
               *boolean-schema*)))
    (format t "Output: ~A~2%" gbnf))

  (format t "── Person schema (object with required fields) ──~%")
  (format t "Input:  ~A~%" *person-schema*)
  (let ((gbnf (cl-llama-cpp/common/json-schema:json-schema-to-grammar
               *person-schema*)))
    (format t "Output (~D chars):~%~A~2%" (length gbnf) gbnf))

  (format t "── Sentiment schema (enum + number) ──~%")
  (format t "Input:  ~A~%" *sentiment-schema*)
  (let ((gbnf (cl-llama-cpp/common/json-schema:json-schema-to-grammar
               *sentiment-schema*)))
    (format t "Output (~D chars):~%~A~2%" (length gbnf) gbnf)))

;;; ── Demo 2: Constrained generation ─────────────────────────────────

(defun demo-constrained-generation (model ctx)
  (banner "DEMO 2: Constrained Generation with JSON Schema")

  (format t "Two ways to use JSON Schema with generate:~%")
  (format t "  2a: Convert to GBNF, pass :grammar to generate (recommended)~%")
  (format t "  2b: make-json-schema-sampler + manual chain (lower-level)~2%")

  ;; 2a — The easy way: convert to GBNF, pass :grammar
  (format t "── 2a: generate with :grammar (recommended) ──~%")
  (let ((prompt "Extract the person from this text as JSON: \"Alice is 30 years old and enjoys hiking, reading, and cooking.\"")
        (gbnf (cl-llama-cpp/common/json-schema:json-schema-to-grammar
               *person-schema*)))
    (format t "Prompt: ~A~2%" prompt)
    (format t "Schema: ~A~2%" *person-schema*)
    (let ((result (generate ctx prompt
                            :max-tokens 256
                            :temp 0.0
                            :grammar gbnf)))
      (format t "Result: ~A~2%" result)))

  ;; 2b — Lower-level: make-json-schema-sampler + manual chain
  (format t "── 2b: make-json-schema-sampler + manual chain ──~%")
  (format t "make-json-schema-sampler converts the schema and creates a grammar~%")
  (format t "sampler in one step. The chain takes ownership — freeing the chain~%")
  (format t "frees all its samplers.~2%")
  (let ((prompt "Classify the sentiment of this review as JSON: \"The food was absolutely amazing, best meal I've ever had!\""))
    (format t "Prompt: ~A~2%" prompt)
    (format t "Schema: ~A~2%" *sentiment-schema*)
    (let ((gs (cl-llama-cpp/common/json-schema:make-json-schema-sampler
               model *sentiment-schema*)))
      (with-sampler-chain (chain)
        (sampler-chain-add chain gs)
        (sampler-chain-add chain (make-temp-sampler 0.0))
        (sampler-chain-add chain (make-dist-sampler 42))
        (let ((result (generate ctx prompt
                                :max-tokens 128
                                :sampler chain)))
          (format t "Result: ~A~2%" result))))))

;;; ── Demo 3: Error handling ─────────────────────────────────────────

(defun demo-error-handling ()
  (banner "DEMO 3: Error Handling — Conditions and Restarts")

  (format t "json-schema-to-grammar signals structured conditions on failure.~%")
  (format t "json-schema-parse-error for malformed JSON, and~%")
  (format t "json-schema-conversion-error for schemas that can't be converted.~%")
  (format t "Both are subtypes of cl-llama-cpp:llama-error.~2%")

  ;; Parse error
  (format t "── Malformed JSON ──~%")
  (handler-case
      (cl-llama-cpp/common/json-schema:json-schema-to-grammar "{not valid json}")
    (cl-llama-cpp/common/json-schema:json-schema-parse-error (c)
      (format t "  Caught: ~A~%" c)
      (format t "  Type:   ~A~%"  (type-of c))
      (format t "  Schema: ~S~2%"
              (cl-llama-cpp/common/json-schema:json-schema-conversion-error-schema c))))

  ;; Restart: use-different-schema
  (format t "── Restart: use-different-schema ──~%")
  (format t "  When a schema fails, the use-different-schema restart lets you~%")
  (format t "  supply a corrected schema without unwinding the call stack.~2%")
  (let ((result
          (handler-bind
              ((cl-llama-cpp/common/json-schema:json-schema-parse-error
                 (lambda (c)
                   (declare (ignore c))
                   (format t "  Handler: bad schema detected, retrying with boolean schema~%")
                   (invoke-restart
                    'cl-llama-cpp/common/json-schema::use-different-schema
                    *boolean-schema*))))
            (cl-llama-cpp/common/json-schema:json-schema-to-grammar "oops"))))
    (format t "  Recovered GBNF: ~A~2%" result))

  (format t "Errors are caught cleanly — no crashes or leaks.~%"))

;;; ── Entry point ────────────────────────────────────────────────────

(defun run ()
  "Run all JSON Schema demos."
  ;; Demo 1 and 3 need no model
  (demo-conversion)
  (demo-error-handling)

  ;; Demo 2 needs a model
  (unless *model-path*
    (format t "~2%Set *model-path* or export LLAMA_MODEL to run Demo 2 ")
    (format t "(constrained generation).~%")
    (return-from run (values)))

  (format t "~&Loading model: ~A~%" *model-path*)
  (with-backend ()
    (set-log-callback (lambda (level text)
                        (when (>= level 3)
                          (format *error-output* "~a" text))))
    (with-model (model *model-path* :n-gpu-layers 99)
      (with-context (ctx model :n-ctx 2048)
        (demo-constrained-generation model ctx))))

  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  All demos complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
