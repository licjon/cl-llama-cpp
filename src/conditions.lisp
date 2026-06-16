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

(define-condition chat-template-error (llama-error)
  ()
  (:report (lambda (c s)
             (declare (ignore c))
             (format s "Chat template formatting failed"))))

(define-condition lora-load-error (llama-error)
  ((path :initarg :path :reader lora-load-error-path))
  (:report (lambda (c s)
             (format s "Failed to load LoRA adapter from ~S"
                     (lora-load-error-path c)))))

(define-condition lora-apply-error (llama-error)
  ((code :initarg :code :reader lora-apply-error-code))
  (:report (lambda (c s)
             (format s "Failed to apply LoRA adapter (code ~D)"
                     (lora-apply-error-code c)))))

(define-condition session-save-error (llama-error)
  ((path :initarg :path :reader session-save-error-path))
  (:report (lambda (c s)
             (format s "Failed to save session state to ~S"
                     (session-save-error-path c)))))

(define-condition session-load-error (llama-error)
  ((path :initarg :path :reader session-load-error-path))
  (:report (lambda (c s)
             (format s "Failed to load session state from ~S"
                     (session-load-error-path c)))))

(define-condition grammar-error (llama-error)
  ((grammar :initarg :grammar :reader grammar-error-grammar))
  (:report (lambda (c s)
             (format s "Failed to create grammar sampler for grammar of length ~D"
                     (length (grammar-error-grammar c))))))

(define-condition batch-init-error (llama-error)
  ((n-tokens :initarg :n-tokens :reader batch-init-error-n-tokens))
  (:report (lambda (c s)
             (format s "Failed to allocate batch for ~D tokens"
                     (batch-init-error-n-tokens c)))))

(define-condition batch-overflow-error (llama-error)
  ((capacity :initarg :capacity :reader batch-overflow-error-capacity)
   (token-count :initarg :token-count :reader batch-overflow-error-token-count))
  (:report (lambda (c s)
             (format s "Batch overflow: ~D tokens already at capacity ~D"
                     (batch-overflow-error-token-count c)
                     (batch-overflow-error-capacity c)))))
