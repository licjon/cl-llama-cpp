(defpackage cl-llama-cpp/common/tests/common
  (:use :cl
        :cl-llama-cpp/common
        :rove))
(in-package :cl-llama-cpp/common/tests/common)

;; NOTE: To run this test file, execute `(asdf:test-system :cl-llama-cpp/common)' in your Lisp.

(deftest test-target-1
  (testing "should (= 1 1) to be true"
    (ok (= 1 1))))
