(defpackage #:cl-llama-cpp/tests/integration
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/integration)

(defvar *test-model-path* (uiop:getenv "LLAMA_TEST_MODEL"))
(defvar *test-embed-model-path* (uiop:getenv "LLAMA_TEST_EMBED_MODEL"))

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

(deftest model-chat-template
  (when-model-available
    (testing "model-chat-template retrieves model template"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((tmpl (cl-llama-cpp:model-chat-template model)))
          (ok (or (null tmpl) (stringp tmpl))
              (format nil "template is nil or string: ~s"
                      (if tmpl (subseq tmpl 0 (min 50 (length tmpl))) nil))))))))

(deftest format-chat-basic
  (when-model-available
    (testing "format-chat produces a formatted string"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((result (cl-llama-cpp:format-chat
                       model
                       '((:role "user" :content "Hello")
                         (:role "assistant" :content "Hi there!")
                         (:role "user" :content "How are you?")))))
          (ok (stringp result) "result is a string")
          (ok (> (length result) 0) "result is non-empty")
          (ok (search "Hello" result) "result contains message content"))))))

(deftest embed-text
  (if *test-embed-model-path*
      (testing "embed produces a float vector"
        (cl-llama-cpp:with-model (model *test-embed-model-path* :n-gpu-layers 0)
          (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings 1
                                                :pooling-type 1)
            (let ((embedding (cl-llama-cpp:embed ctx "Hello, world!")))
              (ok (vectorp embedding)
                  "embed returned a vector")
              (ok (> (length embedding) 0)
                  (format nil "embedding has ~d dimensions" (length embedding)))
              (ok (every #'numberp embedding)
                  "all elements are numbers")))))
      (skip "LLAMA_TEST_EMBED_MODEL not set — skipping")))
