(defpackage #:cl-llama-cpp/common/fit/tests
  (:use #:cl #:rove))
(in-package #:cl-llama-cpp/common/fit/tests)

;;; Package and symbol tests

(deftest test-package-exists
  (testing "cl-llama-cpp/common/fit package exists"
    (ok (find-package :cl-llama-cpp/common/fit))))

(deftest test-symbols-exported
  (testing "fit-params is exported and fbound"
    (ok (fboundp (find-symbol "FIT-PARAMS"
                              :cl-llama-cpp/common/fit))))
  (testing "fit-print is exported and fbound"
    (ok (fboundp (find-symbol "FIT-PRINT"
                              :cl-llama-cpp/common/fit))))
  (testing "memory-breakdown-print is exported and fbound"
    (ok (fboundp (find-symbol "MEMORY-BREAKDOWN-PRINT"
                              :cl-llama-cpp/common/fit))))
  (testing "fit-result accessors are exported and fbound"
    (ok (fboundp (find-symbol "FIT-RESULT-STATUS"
                              :cl-llama-cpp/common/fit)))
    (ok (fboundp (find-symbol "FIT-RESULT-N-GPU-LAYERS"
                              :cl-llama-cpp/common/fit)))
    (ok (fboundp (find-symbol "FIT-RESULT-N-CTX"
                              :cl-llama-cpp/common/fit)))
    (ok (fboundp (find-symbol "FIT-RESULT-TENSOR-SPLIT"
                              :cl-llama-cpp/common/fit))))
  (testing "condition symbols are exported"
    (ok (find-symbol "FIT-ERROR"
                     :cl-llama-cpp/common/fit))
    (ok (find-symbol "FIT-ERROR-PATH"
                     :cl-llama-cpp/common/fit))))

;;; Condition hierarchy

(deftest test-condition-hierarchy
  (testing "fit-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/fit:fit-error
                  'cl-llama-cpp:llama-error))))

(deftest test-condition-signaling
  (testing "fit-error can be constructed and caught"
    (ok (typep
         (handler-case
             (error 'cl-llama-cpp/common/fit:fit-error
                    :path "/nonexistent.gguf")
           (cl-llama-cpp/common/fit:fit-error (c) c))
         'cl-llama-cpp/common/fit:fit-error))))

(deftest test-condition-path
  (testing "fit-error-path returns the path"
    (let ((c (make-condition
              'cl-llama-cpp/common/fit:fit-error
              :path "/some/model.gguf")))
      (ok (string= "/some/model.gguf"
                    (cl-llama-cpp/common/fit:fit-error-path c))))))

(deftest test-condition-report
  (testing "fit-error prints a readable report"
    (let ((c (make-condition
              'cl-llama-cpp/common/fit:fit-error
              :path "/bad/model.gguf")))
      (ok (search "/bad/model.gguf" (princ-to-string c))))))

;;; fit-params error on nonexistent file

(deftest test-fit-params-nonexistent-file
  (testing "fitting a nonexistent model signals fit-error"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/fit:fit-params
                    "/nonexistent-model-file.gguf")
                 (cl-llama-cpp/common/fit:fit-error (c) c))
               'cl-llama-cpp/common/fit:fit-error))))

;;; fit-print error on nonexistent file

(deftest test-fit-print-nonexistent-file
  (testing "printing fit for a nonexistent model signals fit-error"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/fit:fit-print
                    "/nonexistent-model-file.gguf")
                 (cl-llama-cpp/common/fit:fit-error (c) c))
               'cl-llama-cpp/common/fit:fit-error))))
