(defpackage #:cl-llama-cpp/common/json-schema/tests
  (:use #:cl #:rove))
(in-package #:cl-llama-cpp/common/json-schema/tests)

(deftest test-package-exists
  (testing "cl-llama-cpp/common/json-schema package exists"
    (ok (find-package :cl-llama-cpp/common/json-schema))))

(deftest test-symbols-exported
  (testing "core symbols are exported"
    (ok (fboundp (find-symbol "JSON-SCHEMA-TO-GRAMMAR"
                              :cl-llama-cpp/common/json-schema)))
    (ok (fboundp (find-symbol "MAKE-JSON-SCHEMA-SAMPLER"
                              :cl-llama-cpp/common/json-schema)))
    (ok (macro-function (find-symbol "WITH-JSON-SCHEMA-SAMPLER"
                                     :cl-llama-cpp/common/json-schema)))))

(deftest test-condition-hierarchy
  (testing "json-schema-conversion-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/json-schema:json-schema-conversion-error
                  'cl-llama-cpp:llama-error)))
  (testing "json-schema-parse-error subtypes json-schema-conversion-error"
    (ok (subtypep 'cl-llama-cpp/common/json-schema:json-schema-parse-error
                  'cl-llama-cpp/common/json-schema:json-schema-conversion-error))))

(deftest test-malformed-json-signals-parse-error
  (testing "empty string signals json-schema-parse-error"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/json-schema:json-schema-to-grammar "")
                 (cl-llama-cpp/common/json-schema:json-schema-parse-error (c) c))
               'cl-llama-cpp/common/json-schema:json-schema-parse-error)))
  (testing "invalid JSON signals json-schema-parse-error"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/json-schema:json-schema-to-grammar "{not json}")
                 (cl-llama-cpp/common/json-schema:json-schema-parse-error (c) c))
               'cl-llama-cpp/common/json-schema:json-schema-parse-error))))

(deftest test-simple-schema-produces-gbnf
  (testing "boolean schema produces non-empty GBNF with root rule"
    (let ((gbnf (cl-llama-cpp/common/json-schema:json-schema-to-grammar
                 "{\"type\":\"boolean\"}")))
      (ok (stringp gbnf))
      (ok (plusp (length gbnf)))
      (ok (search "root" gbnf))))
  (testing "object schema produces non-empty GBNF"
    (let ((gbnf (cl-llama-cpp/common/json-schema:json-schema-to-grammar
                 (format nil "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"]}"))))
      (ok (stringp gbnf))
      (ok (plusp (length gbnf)))
      (ok (search "root" gbnf)))))

(deftest test-hash-table-input
  (testing "hash table with boolean schema produces same GBNF as string"
    (let* ((ht (let ((h (make-hash-table :test 'equal)))
                 (setf (gethash "type" h) "boolean")
                 h))
           (gbnf-ht (cl-llama-cpp/common/json-schema:json-schema-to-grammar ht))
           (gbnf-str (cl-llama-cpp/common/json-schema:json-schema-to-grammar
                      "{\"type\":\"boolean\"}")))
      (ok (string= gbnf-ht gbnf-str))))
  (testing "hash table with object schema produces valid GBNF"
    (let* ((props (let ((h (make-hash-table :test 'equal)))
                    (let ((name-ht (make-hash-table :test 'equal)))
                      (setf (gethash "type" name-ht) "string")
                      (setf (gethash "name" h) name-ht))
                    (let ((age-ht (make-hash-table :test 'equal)))
                      (setf (gethash "type" age-ht) "integer")
                      (setf (gethash "age" h) age-ht))
                    h))
           (ht (let ((h (make-hash-table :test 'equal)))
                 (setf (gethash "type" h) "object")
                 (setf (gethash "properties" h) props)
                 (setf (gethash "required" h) (list "name" "age"))
                 h))
           (gbnf (cl-llama-cpp/common/json-schema:json-schema-to-grammar ht)))
      (ok (stringp gbnf))
      (ok (plusp (length gbnf)))
      (ok (search "root" gbnf)))))

(deftest test-use-different-schema-restart
  (testing "use-different-schema restart is established on error"
    (let ((result
            (handler-bind
                ((cl-llama-cpp/common/json-schema:json-schema-parse-error
                   (lambda (c)
                     (declare (ignore c))
                     (let ((restart (find-restart 'cl-llama-cpp/common/json-schema::use-different-schema)))
                       (when restart
                         (invoke-restart restart "{\"type\":\"boolean\"}"))))))
              (cl-llama-cpp/common/json-schema:json-schema-to-grammar "bad json"))))
      (ok (stringp result))
      (ok (plusp (length result))))))

(deftest test-shim-free-null-pointer
  (testing "freeing a null pointer does not crash (cleanup path for null-pointer guard)"
    (ok (progn
          (cl-llama-cpp/common/json-schema::%shim-free (cffi:null-pointer))
          t))))
