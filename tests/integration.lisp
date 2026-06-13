(defpackage #:cl-llama-cpp/tests/integration
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/integration)

;;; Integration tests gated on LLAMA_TEST_MODEL env var.
;;; Run: LLAMA_TEST_MODEL=/path/to/model.gguf ros run -e '(asdf:test-system "cl-llama-cpp/tests")'
