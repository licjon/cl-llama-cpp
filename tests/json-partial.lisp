(defpackage #:cl-llama-cpp/common/json-partial/tests
  (:use #:cl #:rove))
(in-package #:cl-llama-cpp/common/json-partial/tests)

;;; Package and symbol tests

(deftest test-package-exists
  (testing "cl-llama-cpp/common/json-partial package exists"
    (ok (find-package :cl-llama-cpp/common/json-partial))))

(deftest test-symbols-exported
  (testing "json-partial-parse is exported and fbound"
    (ok (fboundp (find-symbol "JSON-PARTIAL-PARSE"
                              :cl-llama-cpp/common/json-partial))))
  (testing "condition symbols are exported"
    (ok (find-symbol "JSON-PARTIAL-ERROR"
                     :cl-llama-cpp/common/json-partial))
    (ok (find-symbol "JSON-PARTIAL-PARSE-ERROR"
                     :cl-llama-cpp/common/json-partial))))

(deftest test-condition-hierarchy
  (testing "json-partial-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/json-partial:json-partial-error
                  'cl-llama-cpp:llama-error)))
  (testing "json-partial-parse-error subtypes json-partial-error"
    (ok (subtypep 'cl-llama-cpp/common/json-partial:json-partial-parse-error
                  'cl-llama-cpp/common/json-partial:json-partial-error))))

;;; Complete JSON (no healing)

(deftest test-complete-json-object
  (testing "complete JSON object parses without healing"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse
         "{\"name\":\"Alice\"}")
      (ok (stringp json))
      (ok (search "Alice" json))
      (ng healed-p)
      (ok (or (null marker) (string= marker ""))))))

(deftest test-complete-json-array
  (testing "complete JSON array parses without healing"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse "[1,2,3]")
      (ok (stringp json))
      (ok (search "1" json))
      (ng healed-p)
      (ok (or (null marker) (string= marker ""))))))

(deftest test-complete-json-string
  (testing "complete JSON string parses without healing"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse "\"hello\"")
      (ok (stringp json))
      (ok (search "hello" json))
      (ng healed-p))))

(deftest test-complete-json-number
  (testing "complete JSON number parses without healing"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse "42")
      (ok (stringp json))
      (ok (string= json "42"))
      (ng healed-p))))

;;; Partial JSON with healing

(deftest test-partial-object-healed
  (testing "partial object is healed with marker"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse
         "{\"name\":\"Ali"
         :healing-marker "$HEAL$")
      (ok (stringp json))
      (ok healed-p)
      (ok (stringp marker))
      (ok (plusp (length marker))))))

(deftest test-partial-array-healed
  (testing "partial array is healed with marker"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse
         "[1,2"
         :healing-marker "$HEAL$")
      (ok (stringp json))
      (ok healed-p)
      (ok (stringp marker))
      (ok (plusp (length marker))))))

(deftest test-partial-nested-object-healed
  (testing "partial nested object is healed"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse
         "{\"user\":{\"name\":\"Al"
         :healing-marker "$HEAL$")
      (ok (stringp json))
      (ok healed-p)
      (ok (stringp marker)))))

(deftest test-partial-object-after-colon-healed
  (testing "partial object stopped after colon is healed"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse
         "{\"key\":"
         :healing-marker "$HEAL$")
      (ok (stringp json))
      (ok healed-p))))

(deftest test-healing-marker-in-output
  (testing "json-dump-marker can locate healing boundary in output"
    (multiple-value-bind (json healed-p marker)
        (cl-llama-cpp/common/json-partial:json-partial-parse
         "{\"name\":\"Ali"
         :healing-marker "$HEAL$")
      (declare (ignore healed-p))
      (ok (stringp marker))
      (ok (plusp (length marker)))
      (ok (search marker json)))))

;;; No healing when marker is empty/nil

(deftest test-empty-healing-marker-fails-on-partial
  (testing "empty healing marker causes parse failure on partial JSON"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/json-partial:json-partial-parse
                    "{\"name\":\"Ali")
                 (cl-llama-cpp/common/json-partial:json-partial-parse-error (c) c))
               'cl-llama-cpp/common/json-partial:json-partial-parse-error))))

(deftest test-nil-healing-marker-fails-on-partial
  (testing "nil healing marker treated as empty — no healing attempted"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/json-partial:json-partial-parse
                    "{\"name\":\"Ali"
                    :healing-marker nil)
                 (cl-llama-cpp/common/json-partial:json-partial-parse-error (c) c))
               'cl-llama-cpp/common/json-partial:json-partial-parse-error))))

;;; Error cases

(deftest test-empty-input-no-marker-signals-error
  (testing "empty input with no healing marker signals parse error"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/json-partial:json-partial-parse "")
                 (cl-llama-cpp/common/json-partial:json-partial-parse-error (c) c))
               'cl-llama-cpp/common/json-partial:json-partial-parse-error))))

(deftest test-garbage-input-signals-error
  (testing "non-JSON garbage signals parse error"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/json-partial:json-partial-parse "not json at all")
                 (cl-llama-cpp/common/json-partial:json-partial-parse-error (c) c))
               'cl-llama-cpp/common/json-partial:json-partial-parse-error))))

;;; Cleanup safety

(deftest test-shim-free-null-pointer
  (testing "freeing a null pointer does not crash"
    (ok (progn
          (cl-llama-cpp/common/json-partial::%json-partial-free (cffi:null-pointer))
          t))))
