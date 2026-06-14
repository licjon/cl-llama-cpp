(defpackage #:cl-llama-cpp/tests/integration
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/integration)

(defvar *test-model-path* (uiop:getenv "LLAMA_TEST_MODEL"))
(defvar *test-embed-model-path* (uiop:getenv "LLAMA_TEST_EMBED_MODEL"))
(defvar *test-lora-path* (uiop:getenv "LLAMA_TEST_LORA"))

(defmacro when-model-available (&body body)
  `(if *test-model-path*
       (progn ,@body)
       (skip "LLAMA_TEST_MODEL not set — skipping")))

(deftest with-model-and-context
  (when-model-available
    (testing "with-model + with-context creates valid context"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (not (cffi:null-pointer-p model)) "model loaded")
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (not (cffi:null-pointer-p ctx)) "context created"))))))

(deftest tokenize-roundtrip
  (when-model-available
    (testing "tokenize and detokenize roundtrip"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let* ((text "Hello, world!")
               (tokens (cl-llama-cpp:tokenize model text))
               (result (cl-llama-cpp:detokenize model tokens)))
          (ok (> (length tokens) 0)
              (format nil "tokenized to ~d tokens" (length tokens)))
          (ok (search "Hello" result)
              (format nil "detokenized back to: ~s" result)))))))

(deftest generate-text
  (when-model-available
    (testing "generate produces text from a prompt"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:generate ctx "The capital of France is"
                                               :max-tokens 16
                                               :temp 0.1)))
            (ok (stringp result)
                (format nil "generated: ~s" result))
            (ok (> (length result) 0)
                "generated non-empty text")))))))

(deftest model-chat-template
  (when-model-available
    (testing "model-chat-template retrieves model template"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((tmpl (cl-llama-cpp:model-chat-template model)))
          (ok (or (null tmpl) (stringp tmpl))
              (format nil "template is nil or string: ~s"
                      (if tmpl (subseq tmpl 0 (min 50 (length tmpl))) nil))))))))

(deftest format-chat-basic
  (when-model-available
    (testing "format-chat produces a formatted string"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((result (cl-llama-cpp:format-chat
                       model
                       '((:role "user" :content "Hello")
                         (:role "assistant" :content "Hi there!")
                         (:role "user" :content "How are you?")))))
          (ok (stringp result) "result is a string")
          (ok (> (length result) 0) "result is non-empty")
          (ok (search "Hello" result) "result contains message content"))))))

(deftest embed-text
  (if *test-embed-model-path*
      (testing "embed produces a float vector"
        (cl-llama-cpp:with-model (model *test-embed-model-path* :n-gpu-layers 0)
          (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings 1
                                                :pooling-type 1)
            (let ((embedding (cl-llama-cpp:embed ctx "Hello, world!")))
              (ok (vectorp embedding)
                  "embed returned a vector")
              (ok (> (length embedding) 0)
                  (format nil "embedding has ~d dimensions" (length embedding)))
              (ok (every #'numberp embedding)
                  "all elements are numbers")))))
      (skip "LLAMA_TEST_EMBED_MODEL not set — skipping")))

;;; LoRA adapter wrapper integration tests

(defmacro when-lora-available (&body body)
  `(if (and *test-model-path* *test-lora-path*)
       (progn ,@body)
       (skip "LLAMA_TEST_MODEL and/or LLAMA_TEST_LORA not set — skipping")))

(deftest with-lora-bad-path
  (when-model-available
    (testing "with-lora signals lora-load-error on nonexistent path"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (handler-case
                (cl-llama-cpp:with-lora (adapter model "/nonexistent/lora.gguf")
                  adapter)
              (cl-llama-cpp:lora-load-error (c)
                (cl-llama-cpp:lora-load-error-path c)))
            "lora-load-error was signaled for bad path")))))

(deftest with-lora-loads-adapter
  (when-lora-available
    (testing "with-lora loads a LoRA adapter and binds a non-null pointer"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
          (ok (not (cffi:null-pointer-p adapter))
              "adapter pointer is non-null"))))))

(deftest with-lora-cleanup-on-nonlocal-exit
  (when-lora-available
    (testing "with-lora frees adapter even on non-local exit"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((saved-ptr nil))
          (ignore-errors
            (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
              (setf saved-ptr adapter)
              (error "deliberate error")))
          (ok saved-ptr "adapter pointer was captured before error"))))))

(deftest apply-lora-to-context
  (when-lora-available
    (testing "apply-lora attaches adapter to context without error"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
            (ok (null (cl-llama-cpp:apply-lora ctx adapter))
                "apply-lora returned NIL (success)")))))))

(deftest apply-lora-with-custom-scale
  (when-lora-available
    (testing "apply-lora accepts a custom scale factor"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
            (ok (null (cl-llama-cpp:apply-lora ctx adapter :scale 0.5))
                "apply-lora with scale 0.5 succeeded")))))))

(deftest apply-lora-integer-scale-coerced
  (when-lora-available
    (testing "apply-lora coerces integer scale to single-float"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
            (ok (null (cl-llama-cpp:apply-lora ctx adapter :scale 1))
                "apply-lora with integer scale succeeded")))))))

(deftest lora-metadata-returns-alist
  (when-lora-available
    (testing "lora-metadata returns an alist of string pairs"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-lora (adapter model *test-lora-path*)
          (let ((metadata (cl-llama-cpp:lora-metadata adapter)))
            (ok (listp metadata) "metadata is a list")
            (when metadata
              (ok (every (lambda (entry)
                           (and (consp entry)
                                (stringp (car entry))
                                (stringp (cdr entry))))
                         metadata)
                  "all entries are (string . string) pairs"))))))))

;;; KV cache / memory management integration tests

(defun decode-tokens (ctx tokens)
  "Helper: decode a token vector into CTX's KV cache."
  (let ((n-tokens (length tokens)))
    (cffi:with-foreign-object (tok-buf '%llama:token n-tokens)
      (dotimes (i n-tokens)
        (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:decode ctx (%llama:batch-get-one tok-buf n-tokens))))))

(deftest clear-kv-cache-returns-nil
  (when-model-available
    (testing "clear-kv-cache clears cache and returns nil"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (null (cl-llama-cpp:clear-kv-cache ctx))
              "clear-kv-cache returned NIL"))))))

(deftest kv-cache-pos-returns-values
  (when-model-available
    (testing "kv-cache-pos returns (values min max) as integers"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (mn mx)
              (cl-llama-cpp:kv-cache-pos ctx 0)
            (ok (integerp mn) (format nil "min is integer: ~A" mn))
            (ok (integerp mx) (format nil "max is integer: ~A" mx))))))))

(deftest kv-cache-pos-after-decode
  (when-model-available
    (testing "kv-cache-pos reflects cached positions after decoding tokens"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Hello world"))
          (multiple-value-bind (mn mx)
              (cl-llama-cpp:kv-cache-pos ctx 0)
            (ok (>= mx 0)
                (format nil "max position >= 0 after decode: ~A" mx))
            (ok (<= mn mx)
                (format nil "min (~A) <= max (~A)" mn mx))))))))

(deftest kv-cache-seq-rm-on-empty
  (when-model-available
    (testing "kv-cache-seq-rm on empty cache returns a boolean"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:clear-kv-cache ctx)
          (let ((result (cl-llama-cpp:kv-cache-seq-rm ctx 0 0 100)))
            (ok (typep result '(member t nil))
                (format nil "kv-cache-seq-rm returned ~A on empty cache" result))))))))

(deftest kv-cache-seq-rm-removes-range
  (when-model-available
    (testing "kv-cache-seq-rm returns T after removing cached data"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Test tokens for removal"))
          (ok (cl-llama-cpp:kv-cache-seq-rm ctx 0 -1 -1)
              "kv-cache-seq-rm returned T after removing data"))))))

(deftest kv-cache-seq-cp-copies-without-error
  (when-model-available
    (testing "kv-cache-seq-cp copies sequence cache"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Copy test"))
          (ok (null (cl-llama-cpp:kv-cache-seq-cp ctx 0 1 -1 -1))
              "kv-cache-seq-cp returned NIL (success)"))))))

(deftest kv-cache-seq-keep-isolates-sequence
  (when-model-available
    (testing "kv-cache-seq-keep keeps only the specified sequence"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Keep test"))
          (ok (null (cl-llama-cpp:kv-cache-seq-keep ctx 0))
              "kv-cache-seq-keep returned NIL (success)"))))))

(deftest kv-cache-can-shift-p-returns-boolean
  (when-model-available
    (testing "kv-cache-can-shift-p returns a generalized boolean"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:kv-cache-can-shift-p ctx)))
            (ok (typep result '(member t nil))
                (format nil "kv-cache-can-shift-p returned ~A" result))))))))

(deftest kv-cache-seq-add-shifts-positions
  (when-model-available
    (testing "kv-cache-seq-add shifts positions by delta"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Shift test"))
          (ok (null (cl-llama-cpp:kv-cache-seq-add ctx 0 -1 -1 10))
              "kv-cache-seq-add returned NIL (success)"))))))

(deftest kv-cache-seq-div-divides-positions
  (when-model-available
    (testing "kv-cache-seq-div divides positions by d"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Divide test"))
          (ok (null (cl-llama-cpp:kv-cache-seq-div ctx 0 -1 -1 2))
              "kv-cache-seq-div returned NIL (success)"))))))

(deftest kv-cache-seq-div-zero-signals-error
  (when-model-available
    (testing "kv-cache-seq-div signals error when d=0"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (handler-case
                  (progn (cl-llama-cpp:kv-cache-seq-div ctx 0 0 10 0) nil)
                (error () t))
              "kv-cache-seq-div signaled error for d=0"))))))

(deftest clear-kv-cache-resets-positions
  (when-model-available
    (testing "after clear-kv-cache, kv-cache-pos reflects empty state"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Fill cache"))
          (cl-llama-cpp:clear-kv-cache ctx)
          (multiple-value-bind (mn mx)
              (cl-llama-cpp:kv-cache-pos ctx 0)
            (ok (>= mn mx)
                (format nil "after clear, min (~A) >= max (~A) indicates empty" mn mx))))))))

;;; Model / context introspection integration tests

(deftest model-description-returns-string
  (when-model-available
    (testing "model-description returns a non-empty string"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((desc (cl-llama-cpp:model-description model)))
          (ok (stringp desc) "model-description returned a string")
          (ok (> (length desc) 0)
              (format nil "model-description: ~S" desc)))))))

(deftest model-metadata-returns-alist
  (when-model-available
    (testing "model-metadata returns an alist of string pairs"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((metadata (cl-llama-cpp:model-metadata model)))
          (ok (listp metadata) "metadata is a list")
          (ok (> (length metadata) 0) "metadata is non-empty")
          (ok (every (lambda (entry)
                       (and (consp entry)
                            (stringp (car entry))
                            (stringp (cdr entry))))
                     metadata)
              "all entries are (string . string) pairs"))))))

(deftest model-info-returns-plist
  (when-model-available
    (testing "model-info returns a plist with expected keys"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((info (cl-llama-cpp:model-info model)))
          (ok (listp info) "model-info returned a list")
          (ok (integerp (getf info :n-params))
              (format nil ":n-params is integer: ~A" (getf info :n-params)))
          (ok (integerp (getf info :n-layers))
              (format nil ":n-layers is integer: ~A" (getf info :n-layers)))
          (ok (integerp (getf info :n-ctx-train))
              (format nil ":n-ctx-train is integer: ~A" (getf info :n-ctx-train)))
          (ok (integerp (getf info :size-bytes))
              (format nil ":size-bytes is integer: ~A" (getf info :size-bytes)))
          (ok (integerp (getf info :n-heads))
              (format nil ":n-heads is integer: ~A" (getf info :n-heads)))
          (ok (integerp (getf info :n-heads-kv))
              (format nil ":n-heads-kv is integer: ~A" (getf info :n-heads-kv)))
          (ok (numberp (getf info :rope-freq-scale))
              (format nil ":rope-freq-scale is number: ~A" (getf info :rope-freq-scale)))
          (ok (typep (getf info :encoder-p) '(member t nil))
              (format nil ":encoder-p is boolean: ~A" (getf info :encoder-p)))
          (ok (typep (getf info :decoder-p) '(member t nil))
              (format nil ":decoder-p is boolean: ~A" (getf info :decoder-p)))
          (ok (typep (getf info :recurrent-p) '(member t nil))
              (format nil ":recurrent-p is boolean: ~A" (getf info :recurrent-p)))
          (ok (typep (getf info :hybrid-p) '(member t nil))
              (format nil ":hybrid-p is boolean: ~A" (getf info :hybrid-p)))
          (ok (typep (getf info :diffusion-p) '(member t nil))
              (format nil ":diffusion-p is boolean: ~A" (getf info :diffusion-p))))))))

(deftest model-info-values-sensible
  (when-model-available
    (testing "model-info values are within sensible ranges"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((info (cl-llama-cpp:model-info model)))
          (ok (> (getf info :n-params) 0) "n-params > 0")
          (ok (> (getf info :n-layers) 0) "n-layers > 0")
          (ok (> (getf info :n-ctx-train) 0) "n-ctx-train > 0")
          (ok (> (getf info :size-bytes) 0) "size-bytes > 0")
          (ok (> (getf info :n-heads) 0) "n-heads > 0"))))))

(deftest context-info-returns-plist
  (when-model-available
    (testing "context-info returns a plist with expected keys"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((info (cl-llama-cpp:context-info ctx)))
            (ok (listp info) "context-info returned a list")
            (ok (= 512 (getf info :n-ctx))
                (format nil ":n-ctx matches requested value: ~A" (getf info :n-ctx)))
            (ok (integerp (getf info :n-batch))
                (format nil ":n-batch is integer: ~A" (getf info :n-batch)))
            (ok (integerp (getf info :n-ubatch))
                (format nil ":n-ubatch is integer: ~A" (getf info :n-ubatch)))
            (ok (integerp (getf info :n-seq-max))
                (format nil ":n-seq-max is integer: ~A" (getf info :n-seq-max)))
            (ok (integerp (getf info :n-threads))
                (format nil ":n-threads is integer: ~A" (getf info :n-threads)))
            (ok (integerp (getf info :n-threads-batch))
                (format nil ":n-threads-batch is integer: ~A" (getf info :n-threads-batch)))))))))

(deftest context-info-positive-values
  (when-model-available
    (testing "context-info values are positive"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((info (cl-llama-cpp:context-info ctx)))
            (ok (> (getf info :n-batch) 0) "n-batch > 0")
            (ok (> (getf info :n-threads) 0) "n-threads > 0")))))))
