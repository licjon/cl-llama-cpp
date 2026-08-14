(defpackage #:cl-llama-cpp/common/imatrix-loader/tests
  (:use #:cl #:rove))
(in-package #:cl-llama-cpp/common/imatrix-loader/tests)

;;; Package and symbol tests

(deftest test-package-exists
  (testing "cl-llama-cpp/common/imatrix-loader package exists"
    (ok (find-package :cl-llama-cpp/common/imatrix-loader))))

(deftest test-symbols-exported
  (testing "load-imatrix is exported and fbound"
    (ok (fboundp (find-symbol "LOAD-IMATRIX"
                              :cl-llama-cpp/common/imatrix-loader))))
  (testing "struct accessors are exported and fbound"
    (ok (fboundp (find-symbol "IMATRIX-ENTRIES"
                              :cl-llama-cpp/common/imatrix-loader)))
    (ok (fboundp (find-symbol "IMATRIX-DATASETS"
                              :cl-llama-cpp/common/imatrix-loader)))
    (ok (fboundp (find-symbol "IMATRIX-CHUNK-COUNT"
                              :cl-llama-cpp/common/imatrix-loader)))
    (ok (fboundp (find-symbol "IMATRIX-CHUNK-SIZE"
                              :cl-llama-cpp/common/imatrix-loader)))
    (ok (fboundp (find-symbol "IMATRIX-LEGACY-P"
                              :cl-llama-cpp/common/imatrix-loader)))
    (ok (fboundp (find-symbol "IMATRIX-HAS-METADATA-P"
                              :cl-llama-cpp/common/imatrix-loader)))
    (ok (fboundp (find-symbol "IMATRIX-ENTRY-SUMS"
                              :cl-llama-cpp/common/imatrix-loader)))
    (ok (fboundp (find-symbol "IMATRIX-ENTRY-COUNTS"
                              :cl-llama-cpp/common/imatrix-loader))))
  (testing "condition symbols are exported"
    (ok (find-symbol "IMATRIX-LOAD-ERROR"
                     :cl-llama-cpp/common/imatrix-loader))
    (ok (find-symbol "IMATRIX-LOAD-ERROR-FILENAME"
                     :cl-llama-cpp/common/imatrix-loader))))

;;; Condition hierarchy

(deftest test-condition-hierarchy
  (testing "imatrix-load-error subtypes llama-error"
    (ok (subtypep 'cl-llama-cpp/common/imatrix-loader:imatrix-load-error
                  'cl-llama-cpp:llama-error))))

(deftest test-condition-signaling
  (testing "imatrix-load-error can be constructed and caught"
    (ok (typep
         (handler-case
             (error 'cl-llama-cpp/common/imatrix-loader:imatrix-load-error
                    :filename "/nonexistent.dat")
           (cl-llama-cpp/common/imatrix-loader:imatrix-load-error (c) c))
         'cl-llama-cpp/common/imatrix-loader:imatrix-load-error))))

(deftest test-condition-filename
  (testing "imatrix-load-error-filename returns the filename"
    (let ((c (make-condition
              'cl-llama-cpp/common/imatrix-loader:imatrix-load-error
              :filename "/some/path.dat")))
      (ok (string= "/some/path.dat"
                    (cl-llama-cpp/common/imatrix-loader:imatrix-load-error-filename c))))))

(deftest test-condition-report
  (testing "imatrix-load-error prints a readable report"
    (let ((c (make-condition
              'cl-llama-cpp/common/imatrix-loader:imatrix-load-error
              :filename "/bad/file.dat")))
      (ok (search "/bad/file.dat" (princ-to-string c))))))

;;; Load error on nonexistent file

(deftest test-load-nonexistent-file
  (testing "loading a nonexistent file signals imatrix-load-error"
    (ok (typep (handler-case
                   (cl-llama-cpp/common/imatrix-loader:load-imatrix
                    "/nonexistent-imatrix-file.dat")
                 (cl-llama-cpp/common/imatrix-loader:imatrix-load-error (c) c))
               'cl-llama-cpp/common/imatrix-loader:imatrix-load-error))))
