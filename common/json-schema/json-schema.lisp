(in-package #:cl-llama-cpp/common/json-schema)

;;; Conditions

(define-condition json-schema-conversion-error (cl-llama-cpp:llama-error)
  ((schema  :initarg :schema  :reader json-schema-conversion-error-schema)
   (message :initarg :message :reader json-schema-conversion-error-message))
  (:report (lambda (c s)
             (format s "JSON Schema conversion failed: ~A"
                     (json-schema-conversion-error-message c)))))

(define-condition json-schema-parse-error (json-schema-conversion-error)
  ()
  (:report (lambda (c s)
             (format s "JSON Schema parse error: ~A"
                     (json-schema-conversion-error-message c)))))

;;; Core conversion

(defun json-schema-to-grammar (json-schema &key force-gbnf)
  "Convert a JSON Schema to a GBNF grammar string.
JSON-SCHEMA may be a string or a hash table (serialized via yason internally).
Signals JSON-SCHEMA-PARSE-ERROR if the JSON is malformed,
or JSON-SCHEMA-CONVERSION-ERROR if the schema cannot be converted."
  (let ((json-schema-string (etypecase json-schema
                              (string json-schema)
                              (hash-table (with-output-to-string (s)
                                            (yason:encode json-schema s))))))
    (restart-case
        (let ((result (%json-schema-to-grammar
                       json-schema-string
                       (if force-gbnf 1 0))))
          (let ((output-ptr (getf result 'output))
                (status (getf result 'status)))
            (unwind-protect
                 (progn
                   (when (cffi:null-pointer-p output-ptr)
                     (error 'json-schema-conversion-error
                            :schema json-schema-string
                            :message "C shim returned NULL (out of memory?)"))
                   (let ((output-string (cffi:foreign-string-to-lisp output-ptr)))
                     (case status
                       (0 output-string)
                       (1 (error 'json-schema-parse-error
                                 :schema json-schema-string
                                 :message output-string))
                       (otherwise
                        (error 'json-schema-conversion-error
                               :schema json-schema-string
                               :message output-string)))))
              (%shim-free output-ptr))))
      (use-different-schema (new-schema)
        :report "Retry with a different JSON Schema (string or hash table)"
        :interactive (lambda ()
                       (format *query-io* "JSON Schema: ")
                       (list (read-line *query-io*)))
        (json-schema-to-grammar new-schema :force-gbnf force-gbnf)))))

;;; Convenience wrappers that compose with cl-llama-cpp grammar samplers

(defun make-json-schema-sampler (model json-schema
                                 &key force-gbnf (root "root"))
  "Convert JSON Schema to GBNF, then create a grammar sampler.
JSON-SCHEMA may be a string or a hash table.
Composes JSON-SCHEMA-TO-GRAMMAR with CL-LLAMA-CPP:MAKE-GRAMMAR-SAMPLER."
  (let ((gbnf (json-schema-to-grammar json-schema :force-gbnf force-gbnf)))
    (cl-llama-cpp:make-grammar-sampler model gbnf :root root)))

(defmacro with-json-schema-sampler ((var model schema &key force-gbnf (root "root"))
                                    &body body)
  "Create a grammar sampler from a JSON Schema string, bind to VAR, execute BODY, free."
  (let ((gbnf (gensym "GBNF")))
    `(let ((,gbnf (json-schema-to-grammar ,schema :force-gbnf ,force-gbnf)))
       (cl-llama-cpp:with-grammar-sampler (,var ,model ,gbnf :root ,root)
         ,@body))))
