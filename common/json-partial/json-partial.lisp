(in-package #:cl-llama-cpp/common/json-partial)

(define-condition json-partial-error (cl-llama-cpp:llama-error)
  ((message :initarg :message :reader json-partial-error-message))
  (:report (lambda (c s)
             (format s "JSON partial healing error: ~A"
                     (json-partial-error-message c)))))

(define-condition json-partial-parse-error (json-partial-error)
  ()
  (:report (lambda (c s)
             (format s "JSON partial parse failed: ~A"
                     (json-partial-error-message c)))))

(defun json-partial-parse (input &key (healing-marker ""))
  "Parse JSON, optionally healing incomplete JSON by adding missing closers.
HEALING-MARKER controls healing: a non-empty string enables healing of
partial JSON; empty or NIL means no healing is attempted.
Returns three values:
  1. The parsed JSON as a string
  2. T if healing was applied, NIL otherwise
  3. The json-dump-marker string (locates healing boundary in output)"
  (let ((marker (or healing-marker "")))
    (let ((result (%json-partial-parse input marker)))
      (let ((output-ptr (getf result 'json-output))
            (marker-ptr (getf result 'json-dump-marker))
            (status     (getf result 'status)))
        (unwind-protect
             (case status
               (0 (when (cffi:null-pointer-p output-ptr)
                    (error 'json-partial-error
                           :message "shim returned NULL (out of memory?)"))
                  (values (cffi:foreign-string-to-lisp output-ptr) nil ""))
               (1 (when (cffi:null-pointer-p output-ptr)
                    (error 'json-partial-error
                           :message "shim returned NULL (out of memory?)"))
                  (values (cffi:foreign-string-to-lisp output-ptr)
                          t
                          (if (cffi:null-pointer-p marker-ptr)
                              ""
                              (cffi:foreign-string-to-lisp marker-ptr))))
               (2 (error 'json-partial-parse-error
                         :message "could not parse or heal the input"))
               (3 (error 'json-partial-error
                         :message (if (cffi:null-pointer-p output-ptr)
                                      "unknown C++ exception"
                                      (cffi:foreign-string-to-lisp output-ptr))))
               (otherwise
                (error 'json-partial-error
                       :message (format nil "unexpected shim status ~D" status))))
          (unless (cffi:null-pointer-p output-ptr)
            (%json-partial-free output-ptr))
          (unless (cffi:null-pointer-p marker-ptr)
            (%json-partial-free marker-ptr)))))))
