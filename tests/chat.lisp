(defpackage #:cl-llama-cpp/common/chat/tests
  (:use #:cl #:rove))
(in-package #:cl-llama-cpp/common/chat/tests)

;;; Package and symbol tests

(deftest test-package-exists
  (testing "cl-llama-cpp/common/chat package exists"
    (ok (find-package :cl-llama-cpp/common/chat))))

(deftest test-symbols-exported
  (testing "lifecycle functions are exported and fbound"
    (ok (fboundp (find-symbol "MAKE-CHAT-TEMPLATES"
                              :cl-llama-cpp/common/chat)))
    (ok (fboundp (find-symbol "FREE-CHAT-TEMPLATES"
                              :cl-llama-cpp/common/chat)))
    (ok (macro-function (find-symbol "WITH-CHAT-TEMPLATES"
                                     :cl-llama-cpp/common/chat))))
  (testing "core API functions are exported and fbound"
    (ok (fboundp (find-symbol "CHAT-TEMPLATES-APPLY"
                              :cl-llama-cpp/common/chat)))
    (ok (fboundp (find-symbol "CHAT-PARSE"
                              :cl-llama-cpp/common/chat)))
    (ok (fboundp (find-symbol "CHAT-FORMAT-SINGLE"
                              :cl-llama-cpp/common/chat)))
    (ok (fboundp (find-symbol "CHAT-VERIFY-TEMPLATE"
                              :cl-llama-cpp/common/chat))))
  (testing "condition symbols are exported"
    (ok (find-symbol "CHAT-INIT-ERROR"
                     :cl-llama-cpp/common/chat))
    (ok (find-symbol "CHAT-INIT-ERROR-MESSAGE"
                     :cl-llama-cpp/common/chat))
    (ok (find-symbol "CHAT-PARSE-ERROR"
                     :cl-llama-cpp/common/chat))
    (ok (find-symbol "CHAT-PARSE-ERROR-MESSAGE"
                     :cl-llama-cpp/common/chat))))

;;; Condition hierarchy

(deftest test-condition-hierarchy
  (testing "chat-init-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/chat:chat-init-error
                  'cl-llama-cpp:llama-error)))
  (testing "chat-parse-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/chat:chat-parse-error
                  'cl-llama-cpp:llama-error))))

;;; Condition signaling

(deftest test-chat-init-error-signaling
  (testing "chat-init-error can be constructed and caught"
    (ok (typep
         (handler-case
             (error 'cl-llama-cpp/common/chat:chat-init-error
                    :message "test error")
           (cl-llama-cpp/common/chat:chat-init-error (c) c))
         'cl-llama-cpp/common/chat:chat-init-error))))

(deftest test-chat-init-error-message
  (testing "chat-init-error-message returns the message"
    (let ((c (make-condition
              'cl-llama-cpp/common/chat:chat-init-error
              :message "test msg")))
      (ok (string= "test msg"
                    (cl-llama-cpp/common/chat:chat-init-error-message c))))))

(deftest test-chat-parse-error-signaling
  (testing "chat-parse-error can be constructed and caught"
    (ok (typep
         (handler-case
             (error 'cl-llama-cpp/common/chat:chat-parse-error
                    :message "parse failed")
           (cl-llama-cpp/common/chat:chat-parse-error (c) c))
         'cl-llama-cpp/common/chat:chat-parse-error))))

(deftest test-chat-parse-error-message
  (testing "chat-parse-error-message returns the message"
    (let ((c (make-condition
              'cl-llama-cpp/common/chat:chat-parse-error
              :message "bad parse")))
      (ok (string= "bad parse"
                    (cl-llama-cpp/common/chat:chat-parse-error-message c))))))

;;; chat-verify-template (no model needed — pure string validation)

(deftest test-verify-template-valid
  (testing "valid Jinja chat template returns T"
    (ok (eq t (cl-llama-cpp/common/chat:chat-verify-template
               "{% for message in messages %}{{ message['content'] }}{% endfor %}"
               :use-jinja t)))))

(deftest test-verify-template-invalid
  (testing "invalid template string returns NIL"
    (ok (null (cl-llama-cpp/common/chat:chat-verify-template
               ""
               :use-jinja t)))))

(deftest test-verify-template-empty-string
  (testing "empty string returns NIL"
    (ok (null (cl-llama-cpp/common/chat:chat-verify-template
               ""
               :use-jinja nil)))))

;;; chat-parse (pure parsing, no model needed)

(deftest test-chat-parse-content-only
  (testing "plain text parses to a message with content"
    (let ((msg (cl-llama-cpp/common/chat:chat-parse
                "Hello, world!"
                :format :content-only)))
      (ok (stringp (getf msg :content)))
      (ok (string= "Hello, world!" (getf msg :content)))
      (ok (null (getf msg :tool-calls))))))

(deftest test-chat-parse-partial
  (testing "partial parse with is-partial=t does not signal error"
    (let ((msg (cl-llama-cpp/common/chat:chat-parse
                "Hello"
                :is-partial t
                :format :content-only)))
      (ok (stringp (getf msg :content))))))

(deftest test-chat-parse-empty-input
  (testing "empty string parses to message with empty content"
    (let ((msg (cl-llama-cpp/common/chat:chat-parse
                ""
                :format :content-only)))
      (ok (stringp (getf msg :content))))))

;;; Format constants

(deftest test-format-constants
  (testing "chat format constants are defined"
    (ok (boundp (find-symbol "+CHAT-FORMAT-CONTENT-ONLY+"
                             :cl-llama-cpp/common/chat)))
    (ok (= 0 (symbol-value
              (find-symbol "+CHAT-FORMAT-CONTENT-ONLY+"
                           :cl-llama-cpp/common/chat))))))

;;; Tool choice constants

(deftest test-tool-choice-constants
  (testing "tool choice constants are defined"
    (ok (boundp (find-symbol "+TOOL-CHOICE-AUTO+"
                             :cl-llama-cpp/common/chat)))
    (ok (boundp (find-symbol "+TOOL-CHOICE-REQUIRED+"
                             :cl-llama-cpp/common/chat)))
    (ok (boundp (find-symbol "+TOOL-CHOICE-NONE+"
                             :cl-llama-cpp/common/chat)))))

;;; Lifecycle (free idempotency — tested without a model via null-pointer guard)

(deftest test-free-null-safety
  (testing "freeing a null pointer via shim-free does not crash"
    (ok (progn
          (cl-llama-cpp/common/chat::%shim-free (cffi:null-pointer))
          t))))
