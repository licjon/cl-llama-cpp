(defpackage #:cl-llama-cpp/common/ngram-map/tests
  (:use #:cl #:rove))
(in-package #:cl-llama-cpp/common/ngram-map/tests)

;;; Package and symbol tests

(deftest test-package-exists
  (testing "cl-llama-cpp/common/ngram-map package exists"
    (ok (find-package :cl-llama-cpp/common/ngram-map))))

(deftest test-symbols-exported
  (testing "simple draft function is exported and fbound"
    (ok (fboundp (find-symbol "NGRAM-SIMPLE-DRAFT"
                              :cl-llama-cpp/common/ngram-map))))
  (testing "map lifecycle functions are exported and fbound"
    (ok (fboundp (find-symbol "MAKE-NGRAM-MAP"
                              :cl-llama-cpp/common/ngram-map)))
    (ok (fboundp (find-symbol "FREE-NGRAM-MAP"
                              :cl-llama-cpp/common/ngram-map)))
    (ok (macro-function (find-symbol "WITH-NGRAM-MAP"
                                     :cl-llama-cpp/common/ngram-map))))
  (testing "map operation functions are exported and fbound"
    (ok (fboundp (find-symbol "NGRAM-MAP-BEGIN"
                              :cl-llama-cpp/common/ngram-map)))
    (ok (fboundp (find-symbol "NGRAM-MAP-DRAFT"
                              :cl-llama-cpp/common/ngram-map)))
    (ok (fboundp (find-symbol "NGRAM-MAP-ACCEPT"
                              :cl-llama-cpp/common/ngram-map))))
  (testing "condition symbols are exported"
    (ok (find-symbol "NGRAM-MAP-INIT-ERROR"
                     :cl-llama-cpp/common/ngram-map))
    (ok (find-symbol "NGRAM-MAP-INIT-ERROR-MESSAGE"
                     :cl-llama-cpp/common/ngram-map))))

;;; Condition hierarchy

(deftest test-condition-hierarchy
  (testing "ngram-map-init-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/ngram-map:ngram-map-init-error
                  'cl-llama-cpp:llama-error))))

(deftest test-condition-signaling
  (testing "ngram-map-init-error can be constructed and caught"
    (ok (typep
         (handler-case
             (error 'cl-llama-cpp/common/ngram-map:ngram-map-init-error
                    :message "test error")
           (cl-llama-cpp/common/ngram-map:ngram-map-init-error (c) c))
         'cl-llama-cpp/common/ngram-map:ngram-map-init-error))))

(deftest test-condition-message
  (testing "ngram-map-init-error-message returns the message"
    (let ((c (make-condition
              'cl-llama-cpp/common/ngram-map:ngram-map-init-error
              :message "test msg")))
      (ok (string= "test msg"
                    (cl-llama-cpp/common/ngram-map:ngram-map-init-error-message c))))))

;;; ngram-simple-draft

(deftest test-simple-draft-insufficient-history
  (testing "returns empty vector when token history is too short"
    (let ((result (cl-llama-cpp/common/ngram-map:ngram-simple-draft
                   2 3 #(10 20 30) 40)))
      (ok (vectorp result))
      (ok (zerop (length result))))))

(deftest test-simple-draft-with-pattern
  (testing "finds repeated pattern and returns draft tokens"
    ;; tokens: [10 20 30 40 50 10 20 30], sampled: 40
    ;; pattern (last n-1 tokens + sampled): [30 40]
    ;; match at position 2: tokens[2..3] = [30 40]
    ;; draft = tokens[4..6] = [50 10 20]
    (let ((result (cl-llama-cpp/common/ngram-map:ngram-simple-draft
                   2 3 #(10 20 30 40 50 10 20 30) 40)))
      (ok (vectorp result))
      (ok (= 3 (length result)))
      (ok (= 50 (aref result 0)))
      (ok (= 10 (aref result 1)))
      (ok (= 20 (aref result 2))))))

(deftest test-simple-draft-no-match
  (testing "returns empty vector when no pattern matches"
    (let ((result (cl-llama-cpp/common/ngram-map:ngram-simple-draft
                   2 3 #(10 20 30 40 50 60 70 80) 99)))
      (ok (vectorp result))
      (ok (zerop (length result))))))

;;; ngram-map lifecycle

(deftest test-map-create-free
  (testing "create and free ngram-map without crash"
    (let ((m (cl-llama-cpp/common/ngram-map:make-ngram-map 2 3)))
      (ok (not (null m)))
      (ok (progn
            (cl-llama-cpp/common/ngram-map:free-ngram-map m)
            t)))))

(deftest test-map-free-idempotent
  (testing "double-free of ngram-map does not crash"
    (let ((m (cl-llama-cpp/common/ngram-map:make-ngram-map 2 3)))
      (cl-llama-cpp/common/ngram-map:free-ngram-map m)
      (ok (progn
            (cl-llama-cpp/common/ngram-map:free-ngram-map m)
            t)))))

(deftest test-with-ngram-map
  (testing "with-ngram-map scopes correctly"
    (ok (progn
          (cl-llama-cpp/common/ngram-map:with-ngram-map
              (m 2 3 :key-only-p t :min-hits 1)
            (cl-llama-cpp/common/ngram-map:ngram-map-begin
             m #(10 20 30 40 50 60 70 80)))
          t))))

;;; ngram-map operations

(deftest test-map-draft-no-match
  (testing "returns empty vector when no pattern matches"
    (cl-llama-cpp/common/ngram-map:with-ngram-map
        (m 2 3 :key-only-p t :min-hits 1)
      (let ((tokens #(10 20 30 40 50 60 70 80)))
        (cl-llama-cpp/common/ngram-map:ngram-map-begin m tokens)
        (let ((result (cl-llama-cpp/common/ngram-map:ngram-map-draft
                       m tokens 99)))
          (ok (vectorp result))
          (ok (zerop (length result))))))))

(deftest test-map-draft-with-pattern
  (testing "finds repeated pattern and returns draft tokens"
    (cl-llama-cpp/common/ngram-map:with-ngram-map
        (m 2 3 :key-only-p t :min-hits 1)
      (let ((tokens #(10 20 30 40 50 10 20 30)))
        (cl-llama-cpp/common/ngram-map:ngram-map-begin m tokens)
        (let ((result (cl-llama-cpp/common/ngram-map:ngram-map-draft
                       m tokens 40)))
          (ok (vectorp result))
          (ok (plusp (length result))))))))

(deftest test-map-accept-no-crash
  (testing "accept after draft does not crash"
    (cl-llama-cpp/common/ngram-map:with-ngram-map
        (m 2 3 :key-only-p t :min-hits 1)
      (let ((tokens #(10 20 30 40 50 10 20 30)))
        (cl-llama-cpp/common/ngram-map:ngram-map-begin m tokens)
        (cl-llama-cpp/common/ngram-map:ngram-map-draft m tokens 40)
        (ok (progn
              (cl-llama-cpp/common/ngram-map:ngram-map-accept m 2)
              t))))))

(deftest test-map-accept-without-draft
  (testing "accept without prior draft does not crash"
    (cl-llama-cpp/common/ngram-map:with-ngram-map
        (m 2 3 :key-only-p t :min-hits 1)
      (ok (progn
            (cl-llama-cpp/common/ngram-map:ngram-map-accept m 0)
            t)))))
