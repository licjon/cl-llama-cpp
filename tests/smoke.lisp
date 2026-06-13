(defpackage #:cl-llama-cpp/tests/smoke
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/smoke)

(deftest bindings-package-exists
  (testing "%llama package exists and has symbols"
    (let ((pkg (find-package :%llama)))
      (ok pkg "%llama package exists")
      (let ((count 0))
        (do-symbols (s pkg) (incf count))
        (ok (> count 50) (format nil "%llama has ~d symbols (expected >50)" count))))))

(deftest model-default-params
  (testing "llama_model_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-fp-traps-masked
      (let ((params (%llama:model-default-params)))
        (ok params "model-default-params returned non-nil")
        (ok (listp params) "result is a plist")))))

(deftest context-default-params
  (testing "llama_context_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-fp-traps-masked
      (let ((params (%llama:context-default-params)))
        (ok params "context-default-params returned non-nil")
        (ok (listp params) "result is a plist")))))

(deftest sampler-chain-default-params
  (testing "llama_sampler_chain_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-fp-traps-masked
      (let ((params (%llama:sampler-chain-default-params)))
        (ok (listp params) "result is a plist")))))

(deftest backend-init
  (testing "llama_backend_init runs without error"
    (cl-llama-cpp:with-fp-traps-masked
      (%llama:backend-init)
      (pass "backend-init completed"))))

(deftest condition-hierarchy
  (testing "llama-error condition hierarchy exists"
    (ok (subtypep 'cl-llama-cpp:model-load-error 'cl-llama-cpp:llama-error)
        "model-load-error is a llama-error")
    (ok (subtypep 'cl-llama-cpp:context-creation-error 'cl-llama-cpp:llama-error)
        "context-creation-error is a llama-error")
    (ok (subtypep 'cl-llama-cpp:tokenization-error 'cl-llama-cpp:llama-error)
        "tokenization-error is a llama-error")
    (ok (subtypep 'cl-llama-cpp:decode-error 'cl-llama-cpp:llama-error)
        "decode-error is a llama-error")))

(deftest condition-signaling
  (testing "conditions can be signaled and caught"
    (ok (typep (handler-case (error 'cl-llama-cpp:model-load-error :path "/bad/path")
                (cl-llama-cpp:model-load-error (c) c))
               'cl-llama-cpp:model-load-error)
        "model-load-error is catchable")
    (ok (typep (handler-case (error 'cl-llama-cpp:decode-error :code -1)
                (cl-llama-cpp:decode-error (c) c))
               'cl-llama-cpp:decode-error)
        "decode-error is catchable")))

(deftest with-model-bad-path
  (testing "with-model signals model-load-error on nonexistent path"
    (cl-llama-cpp:with-fp-traps-masked
      (%llama:backend-init)
      (ok (handler-case
              (cl-llama-cpp:with-model (model "/nonexistent/path/to/model.gguf")
                nil)
            (cl-llama-cpp:model-load-error (c)
              (cl-llama-cpp:model-load-error-path c)))
          "model-load-error was signaled for bad path"))))
