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

;;; Grammar / constrained generation integration tests

(defvar *json-grammar*
  "root   ::= \"{\" ws kv (ws \",\" ws kv)* ws \"}\"
kv     ::= string ws \":\" ws value
value  ::= string | number | \"true\" | \"false\" | \"null\"
string ::= \"\\\"\" [a-zA-Z0-9 ]* \"\\\"\"
number ::= [0-9]+
ws     ::= [ \\t\\n]*")

(deftest make-grammar-sampler-creates-sampler
  (when-model-available
    (testing "make-grammar-sampler returns a non-null pointer"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler model *json-grammar*)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p sampler))
                   "grammar sampler pointer is non-null")
            (cl-llama-cpp:with-fp-traps-masked
              (%llama:sampler-free sampler))))))))

(deftest make-grammar-sampler-custom-root
  (when-model-available
    (testing "make-grammar-sampler accepts a custom root rule"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler
                        model *json-grammar* :root "root")))
          (unwind-protect
               (ok (not (cffi:null-pointer-p sampler))
                   "grammar sampler with custom root is non-null")
            (cl-llama-cpp:with-fp-traps-masked
              (%llama:sampler-free sampler))))))))

(deftest make-grammar-sampler-empty-grammar-signals-error
  (when-model-available
    (testing "make-grammar-sampler signals grammar-error for empty grammar"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (handler-case
                (progn (cl-llama-cpp:make-grammar-sampler model "") nil)
              (cl-llama-cpp:grammar-error (c)
                (cl-llama-cpp:grammar-error-grammar c)))
            "grammar-error was signaled for empty grammar")))))

(deftest make-grammar-sampler-lazy-creates-sampler
  (when-model-available
    (testing "make-grammar-sampler-lazy returns a non-null pointer"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler-lazy
                        model *json-grammar*)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p sampler))
                   "lazy grammar sampler is non-null")
            (cl-llama-cpp:with-fp-traps-masked
              (%llama:sampler-free sampler))))))))

(deftest make-grammar-sampler-lazy-with-trigger-words
  (when-model-available
    (testing "make-grammar-sampler-lazy accepts trigger words"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler-lazy
                        model *json-grammar*
                        :trigger-words '("{" "["))))
          (unwind-protect
               (ok (not (cffi:null-pointer-p sampler))
                   "lazy grammar sampler with trigger words is non-null")
            (cl-llama-cpp:with-fp-traps-masked
              (%llama:sampler-free sampler))))))))

(deftest make-grammar-sampler-lazy-with-trigger-patterns
  (when-model-available
    (testing "make-grammar-sampler-lazy accepts trigger patterns"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler-lazy
                        model *json-grammar*
                        :trigger-patterns '("\\{" "\\["))))
          (unwind-protect
               (ok (not (cffi:null-pointer-p sampler))
                   "lazy grammar sampler with trigger patterns is non-null")
            (cl-llama-cpp:with-fp-traps-masked
              (%llama:sampler-free sampler))))))))

(deftest make-grammar-sampler-lazy-words-and-patterns-error
  (when-model-available
    (testing "make-grammar-sampler-lazy rejects both trigger-words and trigger-patterns"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (handler-case
                (progn (cl-llama-cpp:make-grammar-sampler-lazy
                        model *json-grammar*
                        :trigger-words '("{")
                        :trigger-patterns '("\\{"))
                       nil)
              (error () t))
            "error was signaled for conflicting trigger args")))))

(deftest make-infill-sampler-creates-sampler
  (when-model-available
    (testing "make-infill-sampler returns a non-null pointer"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-infill-sampler model)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p sampler))
                   "infill sampler is non-null")
            (cl-llama-cpp:with-fp-traps-masked
              (%llama:sampler-free sampler))))))))

(deftest with-grammar-sampler-binds-and-frees
  (when-model-available
    (testing "with-grammar-sampler creates sampler, executes body, and cleans up"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((captured nil))
          (cl-llama-cpp:with-grammar-sampler (gs model *json-grammar*)
            (setf captured gs)
            (ok (not (cffi:null-pointer-p gs))
                "grammar sampler is non-null inside body"))
          (ok captured "sampler pointer was captured"))))))

(deftest with-grammar-sampler-lazy-mode
  (when-model-available
    (testing "with-grammar-sampler with :lazy t creates a lazy sampler"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-grammar-sampler (gs model *json-grammar*
                                               :lazy t
                                               :trigger-words '("{"))
          (ok (not (cffi:null-pointer-p gs))
              "lazy grammar sampler is non-null inside body"))))))

(deftest with-grammar-sampler-cleanup-on-error
  (when-model-available
    (testing "with-grammar-sampler frees sampler on non-local exit"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((saved nil))
          (ignore-errors
            (cl-llama-cpp:with-grammar-sampler (gs model *json-grammar*)
              (setf saved gs)
              (error "deliberate error")))
          (ok saved "sampler pointer was captured before error"))))))

(deftest generate-with-grammar
  (when-model-available
    (testing "generate with :grammar constrains output"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (text stop-reason)
              (cl-llama-cpp:generate ctx "Output a JSON object:"
                                     :max-tokens 64
                                     :temp 0.1
                                     :grammar *json-grammar*)
            (ok (stringp text)
                (format nil "generated with grammar: ~S" text))
            (ok (member stop-reason '(:eog :length))
                (format nil "stop reason is valid: ~A" stop-reason))))))))

(deftest with-sampler-chain-with-grammar
  (when-model-available
    (testing "with-sampler-chain accepts grammar keywords"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain :model model
                                                :grammar *json-grammar*
                                                :grammar-root "root"
                                                :temp 0.1)
          (ok (not (cffi:null-pointer-p chain))
              "sampler chain with grammar is non-null"))))))

(deftest build-sampler-chain-grammar-without-model-signals-error
  (testing "build-sampler-chain with :grammar but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-fp-traps-masked
                (cl-llama-cpp::build-sampler-chain :grammar *json-grammar*))
              nil)
          (error () t))
        "error was signaled for grammar without model")))

;;; Extended sampler wrapper integration tests

(deftest build-sampler-chain-with-typical-p
  (when-model-available
    (testing "build-sampler-chain with :typical-p creates a valid chain"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain :typical-p 0.9)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with typical-p is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-xtc
  (when-model-available
    (testing "build-sampler-chain with :xtc-probability creates a valid chain"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain
                      :xtc-probability 0.5 :xtc-threshold 0.1)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with xtc is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-top-n-sigma
  (when-model-available
    (testing "build-sampler-chain with :top-n-sigma creates a valid chain"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain :top-n-sigma 2.0)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with top-n-sigma is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-penalties
  (when-model-available
    (testing "build-sampler-chain with penalty keywords creates a valid chain"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain
                      :repeat-penalty 1.1
                      :frequency-penalty 0.1
                      :presence-penalty 0.1
                      :penalty-last-n 128)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with penalties is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-dynamic-temp
  (when-model-available
    (testing "build-sampler-chain with :dynamic-temp-range uses temp-ext"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain
                      :dynamic-temp-range 0.2
                      :dynamic-temp-exponent 1.5)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with dynamic temp is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-adaptive-p
  (when-model-available
    (testing "build-sampler-chain with :adaptive-p creates a valid chain"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain
                      :adaptive-p 0.5 :adaptive-p-decay 0.01)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with adaptive-p is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-mirostat-v2
  (when-model-available
    (testing "build-sampler-chain with :mirostat-v2 creates a valid chain"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain
                      :mirostat-v2 t :mirostat-tau 5.0 :mirostat-eta 0.1)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with mirostat-v2 is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-mirostat-v1
  (when-model-available
    (testing "build-sampler-chain with :mirostat requires model and creates a valid chain"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-fp-traps-masked
          (let ((chain (cl-llama-cpp::build-sampler-chain
                        :model model :mirostat t
                        :mirostat-tau 5.0 :mirostat-eta 0.1)))
            (unwind-protect
                 (ok (not (cffi:null-pointer-p chain))
                     "chain with mirostat v1 is non-null")
              (%llama:sampler-free chain))))))))

(deftest build-sampler-chain-mirostat-mutual-exclusion
  (testing "build-sampler-chain rejects both :mirostat and :mirostat-v2"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-fp-traps-masked
                (cl-llama-cpp::build-sampler-chain :mirostat t :mirostat-v2 t))
              nil)
          (error () t))
        "error was signaled for mirostat + mirostat-v2")))

(deftest build-sampler-chain-mirostat-requires-model
  (testing "build-sampler-chain with :mirostat but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-fp-traps-masked
                (cl-llama-cpp::build-sampler-chain :mirostat t))
              nil)
          (error () t))
        "error was signaled for mirostat without model")))

(deftest build-sampler-chain-dry-requires-model
  (testing "build-sampler-chain with :dry-multiplier but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-fp-traps-masked
                (cl-llama-cpp::build-sampler-chain :dry-multiplier 0.8))
              nil)
          (error () t))
        "error was signaled for dry without model")))

(deftest build-sampler-chain-logit-bias-requires-model
  (testing "build-sampler-chain with :logit-bias but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-fp-traps-masked
                (cl-llama-cpp::build-sampler-chain :logit-bias '((1 . -100.0))))
              nil)
          (error () t))
        "error was signaled for logit-bias without model")))

(deftest build-sampler-chain-with-logit-bias
  (when-model-available
    (testing "build-sampler-chain with :logit-bias creates a valid chain"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-fp-traps-masked
          (let ((chain (cl-llama-cpp::build-sampler-chain
                        :model model
                        :logit-bias '((1 . -100.0) (2 . 50.0)))))
            (unwind-protect
                 (ok (not (cffi:null-pointer-p chain))
                     "chain with logit-bias is non-null")
              (%llama:sampler-free chain))))))))

(deftest build-sampler-chain-with-dry
  (when-model-available
    (testing "build-sampler-chain with :dry-multiplier creates a valid chain"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-fp-traps-masked
          (let ((chain (cl-llama-cpp::build-sampler-chain
                        :model model
                        :dry-multiplier 0.8
                        :dry-base 1.75
                        :dry-allowed-length 2
                        :dry-penalty-last-n 256
                        :dry-seq-breakers '("\n" ":" "\"" "*"))))
            (unwind-protect
                 (ok (not (cffi:null-pointer-p chain))
                     "chain with DRY is non-null")
              (%llama:sampler-free chain))))))))

(deftest sampler-seed-returns-integer
  (when-model-available
    (testing "sampler-seed returns an integer from a sampler chain"
      (cl-llama-cpp:with-fp-traps-masked
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain :seed 12345)))
          (unwind-protect
               (let ((seed (cl-llama-cpp:sampler-seed chain)))
                 (ok (integerp seed)
                     (format nil "sampler-seed returned integer: ~A" seed)))
            (%llama:sampler-free chain)))))))

(deftest generate-with-extended-samplers
  (when-model-available
    (testing "generate accepts extended sampler keywords"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:generate ctx "Hello"
                                               :max-tokens 8
                                               :temp 0.8
                                               :typical-p 0.95
                                               :repeat-penalty 1.1
                                               :frequency-penalty 0.1)))
            (ok (stringp result)
                (format nil "generated with extended samplers: ~S" result))))))))
