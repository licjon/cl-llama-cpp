(defpackage #:cl-llama-cpp/common/speculative/tests
  (:use #:cl #:rove))
(in-package #:cl-llama-cpp/common/speculative/tests)

(deftest test-package-exists
  (testing "cl-llama-cpp/common/speculative package exists"
    (ok (find-package :cl-llama-cpp/common/speculative))))

(deftest test-symbols-exported
  (testing "core symbols are exported and fbound"
    (ok (fboundp (find-symbol "MAKE-SPECULATIVE-PARAMS"
                              :cl-llama-cpp/common/speculative)))
    (ok (fboundp (find-symbol "FREE-SPECULATIVE-PARAMS"
                              :cl-llama-cpp/common/speculative)))
    (ok (fboundp (find-symbol "MAKE-SPECULATIVE-CONTEXT"
                              :cl-llama-cpp/common/speculative)))
    (ok (fboundp (find-symbol "FREE-SPECULATIVE-CONTEXT"
                              :cl-llama-cpp/common/speculative)))
    (ok (fboundp (find-symbol "SPECULATIVE-DRAFT"
                              :cl-llama-cpp/common/speculative)))
    (ok (macro-function
         (find-symbol "WITH-SPECULATIVE-PARAMS"
                      :cl-llama-cpp/common/speculative)))
    (ok (macro-function
         (find-symbol "WITH-SPECULATIVE-CONTEXT"
                      :cl-llama-cpp/common/speculative)))))

(deftest test-type-constants
  (testing "speculative type constants have expected values"
    (ok (= cl-llama-cpp/common/speculative:+speculative-type-none+ 0))
    (ok (= cl-llama-cpp/common/speculative:+speculative-type-draft-simple+ 1))
    (ok (= cl-llama-cpp/common/speculative:+speculative-type-ngram-simple+ 4))))

(deftest test-condition-hierarchy
  (testing "speculative-init-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/speculative:speculative-init-error
                  'cl-llama-cpp:llama-error))))

(deftest test-params-create-free
  (testing "create and free params without crash"
    (let ((p (cl-llama-cpp/common/speculative:make-speculative-params)))
      (ok (not (null p)))
      (ok (progn
            (cl-llama-cpp/common/speculative:free-speculative-params p)
            t)))))

(deftest test-with-speculative-params
  (testing "with-speculative-params scopes correctly"
    (ok (progn
          (cl-llama-cpp/common/speculative:with-speculative-params (p)
            (cl-llama-cpp/common/speculative:speculative-params-add-type
             p cl-llama-cpp/common/speculative:+speculative-type-ngram-simple+)
            (cl-llama-cpp/common/speculative:speculative-params-set-ngram-n p 12)
            (cl-llama-cpp/common/speculative:speculative-params-set-ngram-m p 48)
            (cl-llama-cpp/common/speculative:speculative-params-set-ngram-min-hits p 1))
          t))))

(deftest test-params-free-idempotent
  (testing "double-free of params does not crash"
    (let ((p (cl-llama-cpp/common/speculative:make-speculative-params)))
      (cl-llama-cpp/common/speculative:free-speculative-params p)
      (ok (progn
            (cl-llama-cpp/common/speculative:free-speculative-params p)
            t)))))

(deftest test-n-max-with-ngram
  (testing "speculative-params-n-max returns a non-negative value"
    (cl-llama-cpp/common/speculative:with-speculative-params (p)
      (cl-llama-cpp/common/speculative:speculative-params-add-type
       p cl-llama-cpp/common/speculative:+speculative-type-ngram-simple+)
      (ok (>= (cl-llama-cpp/common/speculative:speculative-params-n-max p) 0)))))
