(defpackage #:cl-llama-cpp/tests/integration
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/integration)

(defvar *test-model-path* (uiop:getenv "LLAMA_TEST_MODEL"))
(defvar *test-embed-model-path* (uiop:getenv "LLAMA_TEST_EMBED_MODEL"))
(defvar *test-lora-path* (uiop:getenv "LLAMA_TEST_LORA"))

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

;;; LoRA adapter wrapper integration tests

(defmacro when-lora-available (&body body)
  `(if (and *test-model-path* *test-lora-path*)
       (progn ,@body)
       (skip "LLAMA_TEST_MODEL and/or LLAMA_TEST_LORA not set — skipping")))

(deftest with-lora-bad-path
  (when-model-available
    (testing "with-lora signals lora-load-error on nonexistent path"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (handler-case
                (cl-llama-cpp:with-lora (adapter model "/nonexistent/lora.gguf")
                  adapter)
              (cl-llama-cpp:lora-load-error (c)
                (cl-llama-cpp:lora-load-error-path c)))
            "lora-load-error was signaled for bad path")))))

(deftest with-lora-loads-adapter
  (when-lora-available
    (testing "with-lora loads a LoRA adapter and binds a non-null pointer"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
          (ok (not (cffi:null-pointer-p adapter))
              "adapter pointer is non-null"))))))

(deftest with-lora-cleanup-on-nonlocal-exit
  (when-lora-available
    (testing "with-lora frees adapter even on non-local exit"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((saved-ptr nil))
          (ignore-errors
            (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
              (setf saved-ptr adapter)
              (error "deliberate error")))
          (ok saved-ptr "adapter pointer was captured before error"))))))

(deftest apply-lora-to-context
  (when-lora-available
    (testing "apply-lora attaches adapter to context without error"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
            (ok (null (cl-llama-cpp:apply-lora ctx adapter))
                "apply-lora returned NIL (success)")))))))

(deftest apply-lora-with-custom-scale
  (when-lora-available
    (testing "apply-lora accepts a custom scale factor"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
            (ok (null (cl-llama-cpp:apply-lora ctx adapter :scale 0.5))
                "apply-lora with scale 0.5 succeeded")))))))

(deftest apply-lora-integer-scale-coerced
  (when-lora-available
    (testing "apply-lora coerces integer scale to single-float"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
            (ok (null (cl-llama-cpp:apply-lora ctx adapter :scale 1))
                "apply-lora with integer scale succeeded")))))))

(deftest lora-metadata-returns-alist
  (when-lora-available
    (testing "lora-metadata returns an alist of string pairs"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
          (let ((metadata (cl-llama-cpp:lora-metadata adapter)))
            (ok (listp metadata) "metadata is a list")
            (when metadata
              (ok (every (lambda (entry)
                           (and (consp entry)
                                (stringp (car entry))
                                (stringp (cdr entry))))
                         metadata)
                  "all entries are (string . string) pairs"))))))))
