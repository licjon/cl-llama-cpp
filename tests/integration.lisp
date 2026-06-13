(defpackage #:cl-llama-cpp/tests/integration
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/integration)

(defvar *test-model-path* (uiop:getenv "LLAMA_TEST_MODEL"))

(defmacro when-model-available (&body body)
  `(if *test-model-path*
       (progn ,@body)
       (skip "LLAMA_TEST_MODEL not set — skipping")))

(deftest with-model-and-context
  (when-model-available
    (testing "with-model + with-context creates valid context"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (not (cffi:null-pointer-p model)) "model loaded")
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (not (cffi:null-pointer-p ctx)) "context created"))))))

(deftest tokenize-roundtrip
  (when-model-available
    (testing "tokenize and detokenize roundtrip"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let* ((text "Hello, world!")
               (tokens (cl-llama-cpp:tokenize model text))
               (result (cl-llama-cpp:detokenize model tokens)))
          (ok (> (length tokens) 0)
              (format nil "tokenized to ~d tokens" (length tokens)))
          (ok (search "Hello" result)
              (format nil "detokenized back to: ~s" result)))))))

(deftest generate-text
  (when-model-available
    (testing "generate produces text from a prompt"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:generate ctx "The capital of France is"
                                               :max-tokens 16
                                               :temp 0.1)))
            (ok (stringp result)
                (format nil "generated: ~s" result))
            (ok (> (length result) 0)
                "generated non-empty text")))))))
