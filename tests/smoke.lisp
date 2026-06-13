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
