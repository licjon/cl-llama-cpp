(in-package #:cl-llama-cpp)

(define-condition llama-error (error)
  ()
  (:documentation "Base condition for cl-llama-cpp errors."))

(define-condition model-load-error (llama-error)
  ((path :initarg :path :reader model-load-error-path))
  (:report (lambda (c s)
             (format s "Failed to load model from ~S" (model-load-error-path c)))))

(define-condition context-creation-error (llama-error)
  ()
  (:report (lambda (c s)
             (declare (ignore c))
             (format s "Failed to create llama context"))))

(define-condition tokenization-error (llama-error)
  ((text :initarg :text :reader tokenization-error-text))
  (:report (lambda (c s)
             (format s "Tokenization failed for text of length ~D"
                     (length (tokenization-error-text c))))))

(define-condition decode-error (llama-error)
  ((code :initarg :code :reader decode-error-code))
  (:report (lambda (c s)
             (format s "llama_decode failed with code ~D" (decode-error-code c)))))
