(defpackage #:cl-llama-cpp/tests/generate
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/generate)

(defun write-test-file (forms)
  "Write FORMS (a list of read forms) to a fresh temp file using %LLAMA
as the reader package, and return its pathname."
  (let ((path (merge-pathnames (format nil "generate-test-~A.lisp" (random 1000000))
                                (uiop:temporary-directory))))
    (with-open-file (out path :direction :output :if-exists :supersede
                              :external-format :utf-8)
      (let ((*package* (find-package :%llama))
            (*print-case* :downcase))
        (dolist (form forms)
          (prin1 form out)
          (terpri out))))
    path))

(deftest extract-definition-forms-basic
  (testing "extracts a defcfun form keyed by its Lisp symbol"
    (let* ((sym (intern "TEST-FOO" :%llama))
           (path (write-test-file
                  (list `(cffi:defcfun ("test_foo" ,sym) :void)))))
      (let ((table (cl-llama-cpp/generate::extract-definition-forms path)))
        (ok (gethash sym table) "TEST-FOO form is present in the table"))
      (delete-file path))))

(deftest report-signature-diff-detects-changed-params
  (testing "flags a function whose parameter list changed between files"
    (let* ((sym (intern "TEST-BAR" :%llama))
           (old-path (write-test-file
                      (list `(cffi:defcfun ("test_bar" ,sym) :void (x :int)))))
           (new-path (write-test-file
                      (list `(cffi:defcfun ("test_bar" ,sym) :void (x :int) (y :int))))))
      (let ((changed (cl-llama-cpp/generate::report-signature-diff old-path new-path)))
        (ok (member sym changed) "TEST-BAR is reported as changed"))
      (delete-file old-path)
      (delete-file new-path))))

(deftest report-signature-diff-ignores-unchanged
  (testing "does not flag a function whose form is identical in both files"
    (let* ((sym (intern "TEST-BAZ" :%llama))
           (form `(cffi:defcfun ("test_baz" ,sym) :void (x :int)))
           (old-path (write-test-file (list form)))
           (new-path (write-test-file (list form))))
      (let ((changed (cl-llama-cpp/generate::report-signature-diff old-path new-path)))
        (ok (not (member sym changed)) "TEST-BAZ is not reported as changed"))
      (delete-file old-path)
      (delete-file new-path))))

(deftest report-signature-diff-ignores-names-not-in-both
  (testing "a symbol only present in one file is not reported as a signature change"
    (let* ((sym (intern "TEST-ONLY-NEW" :%llama))
           (old-path (write-test-file nil))
           (new-path (write-test-file
                      (list `(cffi:defcfun ("test_only_new" ,sym) :void)))))
      (let ((changed (cl-llama-cpp/generate::report-signature-diff old-path new-path)))
        (ok (not (member sym changed))
            "a purely-added symbol is not a signature change (it's covered by report-binding-diff)"))
      (delete-file old-path)
      (delete-file new-path))))

(deftest report-binding-diff-returns-generalized-boolean
  (testing "returns non-nil iff a symbol was added or removed"
    (let* ((sym (intern "TEST-QUX" :%llama))
           (empty-path (write-test-file nil))
           (added-path (write-test-file
                        (list `(common-lisp:export ',sym "%LLAMA")))))
      (ok (cl-llama-cpp/generate::report-binding-diff empty-path added-path)
          "reports non-nil when a symbol was added")
      (ok (not (cl-llama-cpp/generate::report-binding-diff empty-path empty-path))
          "reports nil when nothing changed")
      (delete-file empty-path)
      (delete-file added-path))))

(deftest check-bindings-against-detects-drift-without-writing
  (testing "returns non-nil on drift and never modifies the output file"
    (let* ((sym (intern "TEST-QUUX" :%llama))
           (output-path (write-test-file nil))
           (output-before (uiop:read-file-string output-path))
           (new-path (write-test-file
                      (list `(common-lisp:export ',sym "%LLAMA")))))
      (ok (cl-llama-cpp/generate::check-bindings-against new-path :output output-path)
          "drift is detected")
      (ok (string= output-before (uiop:read-file-string output-path))
          "the committed output file was not modified")
      (delete-file output-path)
      (delete-file new-path))))

(deftest check-bindings-against-no-drift
  (testing "returns nil when the new extraction matches the committed output"
    (let* ((form `(common-lisp:export ',(intern "TEST-CORGE" :%llama) "%LLAMA"))
           (output-path (write-test-file (list form)))
           (new-path (write-test-file (list form))))
      (ok (not (cl-llama-cpp/generate::check-bindings-against new-path :output output-path))
          "no drift is reported")
      (delete-file output-path)
      (delete-file new-path))))
