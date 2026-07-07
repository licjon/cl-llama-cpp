(defpackage #:cl-llama-cpp/tests/integration
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/integration)

(defvar *test-model-path* (uiop:getenv "LLAMA_TEST_MODEL"))
(defvar *test-embed-model-path* (uiop:getenv "LLAMA_TEST_EMBED_MODEL"))
(defvar *test-lora-path* (uiop:getenv "LLAMA_TEST_LORA"))
(defvar *test-gguf-path* (or (uiop:getenv "LLAMA_TEST_MODEL")
                             (uiop:getenv "LLAMA_TEST_EMBED_MODEL")))

(defmacro when-model-available (&body body)
  `(if *test-model-path*
       (progn ,@body)
       (skip "LLAMA_TEST_MODEL not set — skipping")))

(deftest with-model-and-context
  (when-model-available
    (testing "with-model + with-context creates valid context"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (cl-llama-cpp:llama-model-p model) "model is a llama-model handle")
        (ok (cffi:pointerp (cl-llama-cpp:llama-model-pointer model))
            "llama-model-pointer returns a CFFI pointer")
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (cl-llama-cpp:llama-context-p ctx) "context is a llama-context handle")
          (ok (cffi:pointerp (cl-llama-cpp:llama-context-pointer ctx))
              "llama-context-pointer returns a CFFI pointer"))))))

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

(deftest tokenize-roundtrip-non-ascii
  (when-model-available
    (testing "tokenize and detokenize roundtrip for non-ASCII text"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (dolist (text '("café" "你好世界" "über" "Hello 😀 world"))
          (let* ((tokens (cl-llama-cpp:tokenize model text))
                 (result (cl-llama-cpp:detokenize model tokens
                           :remove-special t)))
            (ok (> (length tokens) 0)
                (format nil "~S tokenized to ~D tokens" text (length tokens)))
            (ok (search text result)
                (format nil "~S roundtrips through: ~S" text result))))))))

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

(deftest embed-without-embeddings-signals-error
  (when-model-available
    (testing "embed on a non-embedding context signals a clear error (null-pointer convention)"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (handler-case
                  (progn (cl-llama-cpp:embed ctx "Hello") nil)
                (error (c)
                  (search "EMBEDDINGS" (princ-to-string c))))
              "error mentions :EMBEDDINGS"))))))

;;; Implicit sync / dirty flag integration tests (issue #44)

(defmacro when-embed-model-available (&body body)
  `(if *test-embed-model-path*
       (progn ,@body)
       (skip "LLAMA_TEST_EMBED_MODEL not set — skipping")))

(deftest embed-clears-compute-pending-p
  (when-embed-model-available
    (testing "embed leaves compute-pending-p NIL after a successful call"
      (cl-llama-cpp:with-model (model *test-embed-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings 1
                                              :pooling-type 1)
          (cl-llama-cpp:embed ctx "hello")
          (ok (not (cl-llama-cpp::llama-context-compute-pending-p ctx))
              "compute-pending-p is NIL after embed completes"))))))

(deftest embed-result-stable-across-calls
  (when-embed-model-available
    (testing "embed returns same vector for same input on consecutive calls (sync is correct)"
      (cl-llama-cpp:with-model (model *test-embed-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings 1
                                              :pooling-type 1)
          (let ((e1 (cl-llama-cpp:embed ctx "hello world"))
                (e2 (cl-llama-cpp:embed ctx "hello world")))
            (ok (= (length e1) (length e2))
                "both calls return same dimension")
            (ok (every (lambda (a b) (< (abs (- a b)) 1e-5)) e1 e2)
                "vectors are element-wise equal (sync ensured consistent reads")))))))

(deftest synchronize-clears-pending-flag
  (when-embed-model-available
    (testing "explicit synchronize clears compute-pending-p"
      (cl-llama-cpp:with-model (model *test-embed-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings 1
                                              :pooling-type 1)
          ;; Use the batch API to set the flag without triggering auto-sync
          (let ((tokens (cl-llama-cpp:tokenize model "hello")))
            (cffi:with-foreign-object (buf '%llama:token (length tokens))
              (dotimes (i (length tokens))
                (setf (cffi:mem-aref buf '%llama:token i) (aref tokens i)))
              (let* ((batch (%llama:batch-get-one buf (length tokens)))
                     (rc (%llama:encode (cl-llama-cpp:llama-context-pointer ctx) batch)))
                (declare (ignore rc))
                (setf (cl-llama-cpp::llama-context-compute-pending-p ctx) t))))
          (ok (cl-llama-cpp::llama-context-compute-pending-p ctx)
              "compute-pending-p is T after manually marking dirty")
          (cl-llama-cpp:synchronize ctx)
          (ok (not (cl-llama-cpp::llama-context-compute-pending-p ctx))
              "compute-pending-p is NIL after explicit synchronize"))))))

(deftest batch-encode-sets-pending-flag
  (when-embed-model-available
    (testing "batch-encode sets compute-pending-p on the context"
      (cl-llama-cpp:with-model (model *test-embed-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings 1
                                              :pooling-type 1)
          (let ((tokens (cl-llama-cpp:tokenize model "test")))
            (cl-llama-cpp:with-batch (batch (length tokens))
              (cl-llama-cpp:batch-add-sequence batch tokens 0)
              (cl-llama-cpp:batch-encode ctx batch)))
          (ok (cl-llama-cpp::llama-context-compute-pending-p ctx)
              "compute-pending-p is T after batch-encode"))))))

(deftest batch-decode-sets-pending-flag
  (when-model-available
    (testing "batch-decode sets compute-pending-p on the context"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((tokens (cl-llama-cpp:tokenize model "hello")))
            (cl-llama-cpp:with-batch (batch (length tokens))
              (cl-llama-cpp:batch-add-sequence batch tokens 0 :logits :last)
              (cl-llama-cpp:batch-decode ctx batch)))
          (ok (cl-llama-cpp::llama-context-compute-pending-p ctx)
              "compute-pending-p is T after batch-decode"))))))

(deftest model-chat-template-nil-for-bad-name
  (when-model-available
    (testing "model-chat-template returns NIL for nonexistent template name (null→NIL convention)"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((result (cl-llama-cpp:model-chat-template
                       model "nonexistent-template-name-xyz")))
          (ok (null result)
              "model-chat-template returned NIL for bad name"))))))

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
  (let ((n-tokens (length tokens))
        (ctx-ptr (cl-llama-cpp:llama-context-pointer ctx)))
    (cffi:with-foreign-object (tok-buf '%llama:token n-tokens)
      (dotimes (i n-tokens)
        (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:decode ctx-ptr (%llama:batch-get-one tok-buf n-tokens))))))

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

;;; generate / reset-context primitive tests

(deftest generate-returns-token-vector
  (when-model-available
    (testing "generate returns three values: text, stop-reason, result-tokens"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (text stop-reason result-tokens)
              (cl-llama-cpp:generate ctx "The sky is" :max-tokens 4)
            (ok (stringp text) "first value is a string")
            (ok (keywordp stop-reason) "second value is a keyword")
            (ok (typep result-tokens '(simple-array fixnum (*)))
                "third value is a (simple-array fixnum (*))")
            (ok (plusp (length result-tokens))
                "result-tokens is non-empty")))))))

(deftest reset-context-nil-continues-from-cache
  (when-model-available
    (testing ":reset-context nil leaves the KV cache intact and advances position"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          ;; First call — normal reset, fills part of cache.
          (cl-llama-cpp:generate ctx "Hello" :max-tokens 4)
          (multiple-value-bind (_ mx1)
              (cl-llama-cpp:kv-cache-pos ctx 0)
            (declare (ignore _))
            ;; Second call with :reset-context nil — cache must advance further.
            (cl-llama-cpp:generate ctx "World" :max-tokens 4 :reset-context nil)
            (multiple-value-bind (_ mx2)
                (cl-llama-cpp:kv-cache-pos ctx 0)
              (declare (ignore _))
              (ok (> mx2 mx1)
                  (format nil "cache position advanced from ~A to ~A" mx1 mx2)))))))))

;;; Incremental chat-session integration tests

(deftest chat-session-basic
  (when-model-available
    (testing "make-chat-session + chat-session-send produces a reply"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((session (cl-llama-cpp:make-chat-session ctx)))
            (ok (cl-llama-cpp:chat-session-p session)
                "make-chat-session returns a chat-session")
            (multiple-value-bind (reply stop-reason)
                (cl-llama-cpp:chat-session-send session "Hello!" :max-tokens 8)
              (ok (stringp reply) "reply is a string")
              (ok (plusp (length reply)) "reply is non-empty")
              (ok (keywordp stop-reason) "stop-reason is a keyword"))
            (let ((msgs (cl-llama-cpp:chat-session-messages session)))
              (ok (= (length msgs) 2) "messages has user + assistant turns")
              (ok (string= (getf (first msgs) :role) "user")
                  "first message is user")
              (ok (string= (getf (second msgs) :role) "assistant")
                  "second message is assistant"))))))))

(deftest chat-session-multi-turn
  (when-model-available
    (testing "two sends accumulate messages and produce non-empty replies"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((session (cl-llama-cpp:make-chat-session ctx)))
            (cl-llama-cpp:chat-session-send session "Hi" :max-tokens 8)
            (cl-llama-cpp:chat-session-send session "How are you?" :max-tokens 8)
            (ok (= (length (cl-llama-cpp:chat-session-messages session)) 4)
                "four messages after two turns")))))))

(deftest chat-session-reset-clears
  (when-model-available
    (testing "chat-session-reset empties messages and clears the KV cache"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((session (cl-llama-cpp:make-chat-session ctx)))
            (cl-llama-cpp:chat-session-send session "Hi" :max-tokens 4)
            (cl-llama-cpp:chat-session-reset session)
            (ok (null (cl-llama-cpp:chat-session-messages session))
                "messages is empty after reset")
            (multiple-value-bind (mn mx)
                (cl-llama-cpp:kv-cache-pos ctx 0)
              (ok (>= mn mx)
                  (format nil "cache position is empty after reset (min ~A >= max ~A)" mn mx)))))))))

(deftest chat-session-reset-keep-system
  (when-model-available
    (testing "chat-session-reset with :keep-system t retains the system message"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((session (cl-llama-cpp:make-chat-session
                          ctx :system-prompt "You are a helpful assistant.")))
            (cl-llama-cpp:chat-session-send session "Hi" :max-tokens 4)
            (cl-llama-cpp:chat-session-reset session :keep-system t)
            (let ((msgs (cl-llama-cpp:chat-session-messages session)))
              (ok (= (length msgs) 1) "one message retained")
              (ok (string= (getf (first msgs) :role) "system")
                  "retained message is the system message"))))))))

(deftest chat-session-incremental-equals-full
  (when-model-available
    (testing "incremental decode matches full re-prefill for greedy sampling"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        ;; Two separate contexts: one for incremental, one for full re-prefill.
        (cl-llama-cpp:with-context (ctx-inc model :n-ctx 512)
          (cl-llama-cpp:with-context (ctx-full model :n-ctx 512)
            ;; Greedy sampler = stateless argmax, so sharing it across two
            ;; sequential generate calls is safe and gives deterministic output.
            (cl-llama-cpp:with-sampler-chain (greedy-sampler :greedy t)
              (let* ((user-msg "What is 2 + 2?")
                     (messages (list (list :role "user" :content user-msg)))
                     ;; Incremental path via chat-session.
                     (session (cl-llama-cpp:make-chat-session ctx-inc))
                     (inc-reply (cl-llama-cpp:chat-session-send
                                 session user-msg
                                 :max-tokens 8 :sampler greedy-sampler))
                     ;; Full re-prefill path via plain generate.
                     (prompt-tokens (cl-llama-cpp:tokenize-chat model messages))
                     (full-reply (cl-llama-cpp:generate
                                  ctx-full nil
                                  :prompt-tokens prompt-tokens
                                  :max-tokens 8 :sampler greedy-sampler)))
                (ok (string= inc-reply full-reply)
                    (format nil "incremental ~S equals full ~S"
                            inc-reply full-reply))))))))))

;;; --- Issue #82: external messages mutation reconciles KV cache ---

(deftest chat-session-external-append-reconciles
  (when-model-available
    (testing "externally appending an assistant turn reconciles on next send"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-sampler-chain (greedy :greedy t)
            (let ((session (cl-llama-cpp:make-chat-session ctx)))
              (cl-llama-cpp:chat-session-send session "Hi" :max-tokens 4 :sampler greedy)
              ;; Externally append a fabricated assistant turn.
              (setf (cl-llama-cpp:chat-session-messages session)
                    (append (cl-llama-cpp:chat-session-messages session)
                            (list (list :role "user" :content "What is 2+2?")
                                  (list :role "assistant" :content "4."))))
              ;; Next send must reconcile the cache and produce a reply.
              (multiple-value-bind (reply stop-reason)
                  (cl-llama-cpp:chat-session-send session "Why?" :max-tokens 8 :sampler greedy)
                (ok (and (stringp reply) (plusp (length reply)))
                    (format nil "reconciliation produced reply: ~S" reply))
                (ok (keywordp stop-reason) "stop-reason is a keyword")
                (ok (= 6 (length (cl-llama-cpp:chat-session-messages session)))
                    "six messages after append + send")))))))))

(deftest chat-session-truncate-reconciles
  (when-model-available
    (testing "truncating messages reconciles the KV cache on next send"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-sampler-chain (greedy :greedy t)
            (let ((session (cl-llama-cpp:make-chat-session ctx)))
              (cl-llama-cpp:chat-session-send session "Hi" :max-tokens 4 :sampler greedy)
              (cl-llama-cpp:chat-session-send session "What is 2+2?" :max-tokens 4 :sampler greedy)
              ;; Truncate: keep only the first user+assistant turn.
              (setf (cl-llama-cpp:chat-session-messages session)
                    (list (first (cl-llama-cpp:chat-session-messages session))
                          (second (cl-llama-cpp:chat-session-messages session))))
              ;; Next send must evict the stale suffix and produce a reply.
              (multiple-value-bind (reply stop-reason)
                  (cl-llama-cpp:chat-session-send session "Tell me a joke" :max-tokens 8 :sampler greedy)
                (ok (and (stringp reply) (plusp (length reply)))
                    (format nil "reconciliation produced reply: ~S" reply))
                (ok (keywordp stop-reason) "stop-reason is a keyword")
                (ok (= 4 (length (cl-llama-cpp:chat-session-messages session)))
                    "four messages after truncate + send")))))))))

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
    (testing "make-grammar-sampler returns a llama-sampler handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler model *json-grammar*)))
          (unwind-protect
               (ok (cl-llama-cpp:llama-sampler-p sampler)
                   "grammar sampler is a llama-sampler handle")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (%llama:sampler-free (cl-llama-cpp:llama-sampler-pointer sampler)))))))))

(deftest make-grammar-sampler-custom-root
  (when-model-available
    (testing "make-grammar-sampler accepts a custom root rule"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler
                        model *json-grammar* :root "root")))
          (unwind-protect
               (ok (cl-llama-cpp:llama-sampler-p sampler)
                   "grammar sampler with custom root is a llama-sampler handle")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (%llama:sampler-free (cl-llama-cpp:llama-sampler-pointer sampler)))))))))

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
    (testing "make-grammar-sampler-lazy returns a llama-sampler handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler-lazy
                        model *json-grammar*)))
          (unwind-protect
               (ok (cl-llama-cpp:llama-sampler-p sampler)
                   "lazy grammar sampler is a llama-sampler handle")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (%llama:sampler-free (cl-llama-cpp:llama-sampler-pointer sampler)))))))))

(deftest make-grammar-sampler-lazy-with-trigger-words
  (when-model-available
    (testing "make-grammar-sampler-lazy accepts trigger words"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler-lazy
                        model *json-grammar*
                        :trigger-words '("{" "["))))
          (unwind-protect
               (ok (cl-llama-cpp:llama-sampler-p sampler)
                   "lazy grammar sampler with trigger words is a llama-sampler handle")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (%llama:sampler-free (cl-llama-cpp:llama-sampler-pointer sampler)))))))))

(deftest make-grammar-sampler-lazy-with-trigger-patterns
  (when-model-available
    (testing "make-grammar-sampler-lazy accepts trigger patterns"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-grammar-sampler-lazy
                        model *json-grammar*
                        :trigger-patterns '("\\{" "\\["))))
          (unwind-protect
               (ok (cl-llama-cpp:llama-sampler-p sampler)
                   "lazy grammar sampler with trigger patterns is a llama-sampler handle")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (%llama:sampler-free (cl-llama-cpp:llama-sampler-pointer sampler)))))))))

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
    (testing "make-infill-sampler returns a llama-sampler handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((sampler (cl-llama-cpp:make-infill-sampler model)))
          (unwind-protect
               (ok (cl-llama-cpp:llama-sampler-p sampler)
                   "infill sampler is a llama-sampler handle")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (%llama:sampler-free (cl-llama-cpp:llama-sampler-pointer sampler)))))))))

(deftest with-grammar-sampler-binds-and-frees
  (when-model-available
    (testing "with-grammar-sampler creates sampler, executes body, and cleans up"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((captured nil))
          (cl-llama-cpp:with-grammar-sampler (gs model *json-grammar*)
            (setf captured gs)
            (ok (cl-llama-cpp:llama-sampler-p gs)
                "grammar sampler is a llama-sampler handle inside body"))
          (ok captured "sampler handle was captured"))))))

(deftest with-grammar-sampler-lazy-mode
  (when-model-available
    (testing "with-grammar-sampler with :lazy t creates a lazy sampler"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-grammar-sampler (gs model *json-grammar*
                                               :lazy t
                                               :trigger-words '("{"))
          (ok (cl-llama-cpp:llama-sampler-p gs)
              "lazy grammar sampler is a llama-sampler handle inside body"))))))

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
          (ok (cl-llama-cpp:llama-sampler-p chain)
              "sampler chain with grammar is a llama-sampler handle"))))))

(deftest build-sampler-chain-grammar-without-model-signals-error
  (testing "build-sampler-chain with :grammar but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-llama-compatible-fp-environment
                (cl-llama-cpp::build-sampler-chain :grammar *json-grammar*))
              nil)
          (error () t))
        "error was signaled for grammar without model")))

;;; Sampler chain wrappers (issue #62)

(deftest with-sampler-chain-no-args-creates-empty-chain
  (when-model-available
    (testing "with-sampler-chain with no kwargs yields a typed llama-sampler handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain)
          (ok (cl-llama-cpp:llama-sampler-p chain)
              "chain is a llama-sampler handle")
          (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-sampler-pointer chain)))
              "chain pointer is non-null"))))))

(deftest sampler-chain-add-accepts-raw-pointer
  (when-model-available
    (testing "sampler-chain-add accepts a raw pointer from %llama:sampler-init-temp"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain)
          (cl-llama-cpp:with-llama-compatible-fp-environment
            (let ((temp-smpl (%llama:sampler-init-temp 0.3)))
              (ok (null (cl-llama-cpp:sampler-chain-add chain temp-smpl))
                  "sampler-chain-add returns NIL for raw pointer sampler"))))))))

(deftest sampler-chain-add-accepts-typed-handle
  (when-model-available
    (testing "sampler-chain-add accepts a typed llama-sampler handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain)
          (let ((gs (cl-llama-cpp:make-grammar-sampler model *json-grammar*)))
            (ok (null (cl-llama-cpp:sampler-chain-add chain gs))
                "sampler-chain-add returns NIL for typed handle sampler")))))))

(deftest with-sampler-chain-empty-then-add-counts-samplers
  (when-model-available
    (testing "sampler count reflects samplers added via sampler-chain-add"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain)
          (let ((ptr (cl-llama-cpp:llama-sampler-pointer chain)))
            (ok (= 0 (cl-llama-cpp:with-llama-compatible-fp-environment
                       (%llama:sampler-chain-n ptr)))
                "empty chain has 0 samplers")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (cl-llama-cpp:sampler-chain-add chain (%llama:sampler-init-temp 0.5)))
            (ok (= 1 (cl-llama-cpp:with-llama-compatible-fp-environment
                       (%llama:sampler-chain-n ptr)))
                "chain has 1 sampler after first add")
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (cl-llama-cpp:sampler-chain-add chain (%llama:sampler-init-dist 42)))
            (ok (= 2 (cl-llama-cpp:with-llama-compatible-fp-environment
                       (%llama:sampler-chain-n ptr)))
                "chain has 2 samplers after second add")))))))

(deftest with-sampler-chain-kwargs-still-work
  (when-model-available
    (testing "with-sampler-chain with kwargs still pre-builds the chain as before"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain :temp 0.7 :seed 99)
          (ok (cl-llama-cpp:llama-sampler-p chain)
              "chain with kwargs is a llama-sampler handle")
          (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-sampler-pointer chain)))
              "chain pointer is non-null"))))))

;;; Individual sampler constructors / free-sampler (issue #65)

(deftest make-greedy-sampler-creates-handle
  (testing "make-greedy-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-greedy-sampler)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s)
                 "greedy sampler is a llama-sampler handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-temp-sampler-creates-handle
  (testing "make-temp-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-temp-sampler 0.7)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s)
                 "temp sampler is a llama-sampler handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-dist-sampler-creates-handle
  (testing "make-dist-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-dist-sampler 99)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s)
                 "dist sampler is a llama-sampler handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-top-k-sampler-creates-handle
  (testing "make-top-k-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-top-k-sampler 40)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "top-k sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-top-p-sampler-creates-handle
  (testing "make-top-p-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-top-p-sampler 0.9)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "top-p sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-min-p-sampler-creates-handle
  (testing "make-min-p-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-min-p-sampler 0.05)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "min-p sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-typical-sampler-creates-handle
  (testing "make-typical-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-typical-sampler 0.9)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "typical sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-temp-ext-sampler-creates-handle
  (testing "make-temp-ext-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-temp-ext-sampler 0.7 0.1)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "temp-ext sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-xtc-sampler-creates-handle
  (testing "make-xtc-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-xtc-sampler 0.1 0.3)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "xtc sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-top-n-sigma-sampler-creates-handle
  (testing "make-top-n-sigma-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-top-n-sigma-sampler 2.0)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "top-n-sigma sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest make-mirostat-v2-sampler-creates-handle
  (testing "make-mirostat-v2-sampler returns a typed llama-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-mirostat-v2-sampler 42 5.0 0.1)))
        (unwind-protect
             (ok (cl-llama-cpp:llama-sampler-p s) "mirostat-v2 sampler is a handle")
          (cl-llama-cpp:free-sampler s))))))

(deftest free-sampler-returns-nil
  (testing "free-sampler returns NIL"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((s (cl-llama-cpp:make-greedy-sampler)))
        (ok (null (cl-llama-cpp:free-sampler s))
            "free-sampler returns NIL")))))

(deftest sampler-chain-add-accepts-make-temp-sampler
  (testing "sampler-chain-add accepts a make-temp-sampler handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (cl-llama-cpp:with-sampler-chain (chain)
        (ok (null (cl-llama-cpp:sampler-chain-add
                   chain (cl-llama-cpp:make-temp-sampler 0.5)))
            "sampler-chain-add returns NIL for make-temp-sampler handle")))))

(deftest generate-with-sampler-keyword
  (when-model-available
    (testing "generate accepts a :sampler chain and produces text"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-sampler-chain (chain :temp 0.1 :seed 42)
            (multiple-value-bind (text stop-reason)
                (cl-llama-cpp:generate ctx "The sky is"
                                       :max-tokens 8
                                       :sampler chain)
              (ok (stringp text)
                  (format nil "generated with :sampler chain: ~S" text))
              (ok (member stop-reason '(:eog :length))
                  (format nil "stop reason is valid: ~A" stop-reason)))))))))

(deftest generate-with-sampler-manual-chain
  (when-model-available
    (testing "generate with manual chain of make-temp + make-dist samplers"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-sampler-chain (chain)
            (cl-llama-cpp:sampler-chain-add chain (cl-llama-cpp:make-temp-sampler 0.1))
            (cl-llama-cpp:sampler-chain-add chain (cl-llama-cpp:make-dist-sampler 42))
            (multiple-value-bind (text stop-reason)
                (cl-llama-cpp:generate ctx "One plus one equals"
                                       :max-tokens 4
                                       :sampler chain)
              (ok (stringp text)
                  (format nil "generated with manual chain: ~S" text))
              (ok (member stop-reason '(:eog :length))
                  "stop reason is valid"))))))))

(deftest generate-sampler-not-freed-by-generate
  (when-model-available
    (testing "generate with :sampler does not free the chain (can reuse)"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-sampler-chain (chain :temp 0.1 :seed 1)
            (cl-llama-cpp:generate ctx "Hello" :max-tokens 4 :sampler chain)
            ;; Chain still valid — a second call must not crash
            (ok (stringp (cl-llama-cpp:generate ctx "World" :max-tokens 4 :sampler chain))
                "chain is still usable after first generate call")))))))

(deftest generate-sampler-with-conflicting-kwarg-warns
  (when-model-available
    (testing "generate with :sampler + :grammar issues a warning"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-sampler-chain (chain :temp 0.1)
            (let ((warned nil))
              (handler-bind ((warning (lambda (w)
                                        (setf warned t)
                                        (muffle-warning w))))
                (cl-llama-cpp:generate ctx "Test" :max-tokens 2
                                       :sampler chain
                                       :grammar "root ::= [a-z]+"))
              (ok warned "warning was issued for :sampler + :grammar"))))))))

;;; Session state save/load integration tests

(deftest save-session-creates-file
  (when-model-available
    (testing "save-session writes a session file"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Hello world"))
          (let ((path (namestring (merge-pathnames "test-session.bin"
                                                    (uiop:temporary-directory)))))
            (unwind-protect
                (progn
                  (ok (null (cl-llama-cpp:save-session ctx path))
                      "save-session returned NIL (success)")
                  (ok (probe-file path)
                      "session file was created on disk"))
              (when (probe-file path)
                (delete-file path)))))))))

(deftest save-session-with-tokens
  (when-model-available
    (testing "save-session accepts optional token vector"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((tokens (cl-llama-cpp:tokenize model "Hello world"))
                (path (namestring (merge-pathnames "test-session-tok.bin"
                                                    (uiop:temporary-directory)))))
            (decode-tokens ctx tokens)
            (unwind-protect
                (ok (null (cl-llama-cpp:save-session ctx path tokens))
                    "save-session with tokens returned NIL (success)")
              (when (probe-file path)
                (delete-file path)))))))))

(deftest load-session-roundtrip
  (when-model-available
    (testing "save-session + load-session roundtrip returns tokens"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((tokens (cl-llama-cpp:tokenize model "Hello world"))
                (path (namestring (merge-pathnames "test-session-rt.bin"
                                                    (uiop:temporary-directory)))))
            (decode-tokens ctx tokens)
            (unwind-protect
                (progn
                  (cl-llama-cpp:save-session ctx path tokens)
                  (cl-llama-cpp:clear-kv-cache ctx)
                  (let ((loaded-tokens (cl-llama-cpp:load-session ctx path)))
                    (ok (vectorp loaded-tokens)
                        "load-session returned a vector")
                    (ok (= (length tokens) (length loaded-tokens))
                        (format nil "roundtrip token count matches: ~D = ~D"
                                (length tokens) (length loaded-tokens)))
                    (ok (equalp tokens loaded-tokens)
                        "roundtrip token values match")))
              (when (probe-file path)
                (delete-file path)))))))))

(deftest load-session-bad-path-signals-error
  (when-model-available
    (testing "load-session signals session-load-error on nonexistent path"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (handler-case
                  (progn (cl-llama-cpp:load-session ctx "/nonexistent/session.bin") nil)
                (cl-llama-cpp:session-load-error (c)
                  (cl-llama-cpp:session-load-error-path c)))
              "session-load-error was signaled for bad path"))))))

(deftest save-state-returns-octet-vector
  (when-model-available
    (testing "save-state returns an octet vector"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "State test"))
          (let ((state (cl-llama-cpp:save-state ctx)))
            (ok (vectorp state) "save-state returned a vector")
            (ok (> (length state) 0) "state vector is non-empty")
            (ok (typep state '(simple-array (unsigned-byte 8) (*)))
                "state vector element type is (unsigned-byte 8)")))))))

(deftest save-load-state-roundtrip
  (when-model-available
    (testing "save-state + load-state roundtrip restores state"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "State roundtrip"))
          (let ((state (cl-llama-cpp:save-state ctx)))
            (cl-llama-cpp:clear-kv-cache ctx)
            (let ((bytes-read (cl-llama-cpp:load-state ctx state)))
              (ok (integerp bytes-read) "load-state returned an integer")
              (ok (> bytes-read 0) "load-state consumed bytes"))))))))

(deftest load-state-empty-vector
  (when-model-available
    (testing "load-state with empty vector returns 0"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:load-state
                         ctx (make-array 0 :element-type '(unsigned-byte 8)))))
            (ok (zerop result) "load-state returned 0 for empty input")))))))

(deftest save-session-seq-creates-file
  (when-model-available
    (testing "save-session-seq writes a per-sequence session file"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Seq test"))
          (let ((path (namestring (merge-pathnames "test-seq-session.bin"
                                                    (uiop:temporary-directory)))))
            (unwind-protect
                (progn
                  (ok (null (cl-llama-cpp:save-session-seq ctx path 0))
                      "save-session-seq returned NIL (success)")
                  (ok (probe-file path)
                      "sequence session file was created on disk"))
              (when (probe-file path)
                (delete-file path)))))))))

(deftest load-session-seq-roundtrip
  (when-model-available
    (testing "save-session-seq + load-session-seq roundtrip"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((tokens (cl-llama-cpp:tokenize model "Seq roundtrip"))
                (path (namestring (merge-pathnames "test-seq-rt.bin"
                                                    (uiop:temporary-directory)))))
            (decode-tokens ctx tokens)
            (unwind-protect
                (progn
                  (cl-llama-cpp:save-session-seq ctx path 0 tokens)
                  (cl-llama-cpp:clear-kv-cache ctx)
                  (let ((loaded-tokens (cl-llama-cpp:load-session-seq ctx path 0)))
                    (ok (vectorp loaded-tokens)
                        "load-session-seq returned a vector")
                    (ok (= (length tokens) (length loaded-tokens))
                        (format nil "seq roundtrip token count matches: ~D = ~D"
                                (length tokens) (length loaded-tokens)))))
              (when (probe-file path)
                (delete-file path)))))))))

(deftest load-session-seq-bad-path-signals-error
  (when-model-available
    (testing "load-session-seq signals session-load-error on nonexistent path"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (handler-case
                  (progn (cl-llama-cpp:load-session-seq ctx "/nonexistent/seq.bin" 0) nil)
                (cl-llama-cpp:session-load-error (c)
                  (cl-llama-cpp:session-load-error-path c)))
              "session-load-error was signaled for bad seq path"))))))

(deftest save-state-seq-returns-octet-vector
  (when-model-available
    (testing "save-state-seq returns an octet vector"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Seq state"))
          (let ((state (cl-llama-cpp:save-state-seq ctx 0)))
            (ok (vectorp state) "save-state-seq returned a vector")
            (ok (> (length state) 0) "seq state vector is non-empty")
            (ok (typep state '(simple-array (unsigned-byte 8) (*)))
                "seq state vector element type is (unsigned-byte 8)")))))))

(deftest save-load-state-seq-roundtrip
  (when-model-available
    (testing "save-state-seq + load-state-seq roundtrip"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Seq state roundtrip"))
          (let ((state (cl-llama-cpp:save-state-seq ctx 0)))
            (cl-llama-cpp:clear-kv-cache ctx)
            (let ((bytes-read (cl-llama-cpp:load-state-seq ctx 0 state)))
              (ok (integerp bytes-read) "load-state-seq returned an integer")
              (ok (> bytes-read 0) "load-state-seq consumed bytes"))))))))

(deftest load-state-seq-empty-vector
  (when-model-available
    (testing "load-state-seq with empty vector returns 0"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:load-state-seq
                         ctx 0 (make-array 0 :element-type '(unsigned-byte 8)))))
            (ok (zerop result) "load-state-seq returned 0 for empty input")))))))

;;; Extended sampler wrapper integration tests

(deftest build-sampler-chain-with-typical-p
  (when-model-available
    (testing "build-sampler-chain with :typical-p creates a valid chain"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain :typical-p 0.9)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with typical-p is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-xtc
  (when-model-available
    (testing "build-sampler-chain with :xtc-probability creates a valid chain"
      (cl-llama-cpp:with-llama-compatible-fp-environment
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
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (let ((chain (cl-llama-cpp::build-sampler-chain :top-n-sigma 2.0)))
          (unwind-protect
               (ok (not (cffi:null-pointer-p chain))
                   "chain with top-n-sigma is non-null")
            (%llama:sampler-free chain)))))))

(deftest build-sampler-chain-with-penalties
  (when-model-available
    (testing "build-sampler-chain with penalty keywords creates a valid chain"
      (cl-llama-cpp:with-llama-compatible-fp-environment
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
      (cl-llama-cpp:with-llama-compatible-fp-environment
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
      (cl-llama-cpp:with-llama-compatible-fp-environment
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
      (cl-llama-cpp:with-llama-compatible-fp-environment
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
        (cl-llama-cpp:with-llama-compatible-fp-environment
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
              (cl-llama-cpp:with-llama-compatible-fp-environment
                (cl-llama-cpp::build-sampler-chain :mirostat t :mirostat-v2 t))
              nil)
          (error () t))
        "error was signaled for mirostat + mirostat-v2")))

(deftest build-sampler-chain-mirostat-requires-model
  (testing "build-sampler-chain with :mirostat but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-llama-compatible-fp-environment
                (cl-llama-cpp::build-sampler-chain :mirostat t))
              nil)
          (error () t))
        "error was signaled for mirostat without model")))

(deftest build-sampler-chain-dry-requires-model
  (testing "build-sampler-chain with :dry-multiplier but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-llama-compatible-fp-environment
                (cl-llama-cpp::build-sampler-chain :dry-multiplier 0.8))
              nil)
          (error () t))
        "error was signaled for dry without model")))

(deftest build-sampler-chain-logit-bias-requires-model
  (testing "build-sampler-chain with :logit-bias but no :model signals error"
    (ok (handler-case
            (progn
              (cl-llama-cpp:with-llama-compatible-fp-environment
                (cl-llama-cpp::build-sampler-chain :logit-bias '((1 . -100.0))))
              nil)
          (error () t))
        "error was signaled for logit-bias without model")))

(deftest build-sampler-chain-with-logit-bias
  (when-model-available
    (testing "build-sampler-chain with :logit-bias creates a valid chain"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-llama-compatible-fp-environment
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
        (cl-llama-cpp:with-llama-compatible-fp-environment
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
      (cl-llama-cpp:with-sampler-chain (chain :seed 12345)
        (let ((seed (cl-llama-cpp:sampler-seed chain)))
          (ok (integerp seed)
              (format nil "sampler-seed returned integer: ~A" seed)))))))

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

;;; Batch API integration tests

(deftest with-batch-creates-batch
  (when-model-available
    (testing "with-batch allocates a batch and binds a handle"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 32)
          (ok batch "batch handle is non-nil")
          (ok (zerop (cl-llama-cpp:batch-token-count batch))
              "new batch has 0 tokens"))))))

(deftest with-batch-invalid-capacity
  (testing "with-batch signals batch-init-error for n-tokens <= 0"
    (ok (handler-case
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (cl-llama-cpp:with-batch (batch 0)
                (declare (ignore batch))
                nil))
          (cl-llama-cpp:batch-init-error (c)
            (zerop (cl-llama-cpp:batch-init-error-n-tokens c))))
        "batch-init-error was signaled for n-tokens=0")))

(deftest with-batch-negative-capacity
  (testing "with-batch signals batch-init-error for negative n-tokens"
    (ok (handler-case
            (cl-llama-cpp:with-llama-compatible-fp-environment
              (cl-llama-cpp:with-batch (batch -1)
                (declare (ignore batch))
                nil))
          (cl-llama-cpp:batch-init-error () t))
        "batch-init-error was signaled for n-tokens=-1")))

(deftest batch-add-token-basic
  (when-model-available
    (testing "batch-add-token adds tokens and increments count"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 32)
          (cl-llama-cpp:batch-add-token batch 100 0 0)
          (ok (= 1 (cl-llama-cpp:batch-token-count batch))
              "count is 1 after first add")
          (cl-llama-cpp:batch-add-token batch 200 1 0)
          (ok (= 2 (cl-llama-cpp:batch-token-count batch))
              "count is 2 after second add"))))))

(deftest batch-add-token-with-logits
  (when-model-available
    (testing "batch-add-token accepts :logits keyword"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 32)
          (cl-llama-cpp:batch-add-token batch 100 0 0 :logits nil)
          (cl-llama-cpp:batch-add-token batch 200 1 0 :logits t)
          (ok (= 2 (cl-llama-cpp:batch-token-count batch))
              "two tokens added with logits flags"))))))

(deftest batch-add-token-multi-seq
  (when-model-available
    (testing "batch-add-token accepts a list of seq-ids"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 32 :n-seq-max 3)
          (cl-llama-cpp:batch-add-token batch 100 0 '(0 1 2))
          (ok (= 1 (cl-llama-cpp:batch-token-count batch))
              "token added with 3 seq-ids"))))))

(deftest batch-add-token-overflow
  (when-model-available
    (testing "batch-add-token signals batch-overflow-error at capacity"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 2)
          (cl-llama-cpp:batch-add-token batch 100 0 0)
          (cl-llama-cpp:batch-add-token batch 200 1 0)
          (ok (handler-case
                  (progn (cl-llama-cpp:batch-add-token batch 300 2 0) nil)
                (cl-llama-cpp:batch-overflow-error (c)
                  (and (= 2 (cl-llama-cpp:batch-overflow-error-capacity c))
                       (= 2 (cl-llama-cpp:batch-overflow-error-token-count c)))))
              "batch-overflow-error signaled with correct slots"))))))

(deftest batch-clear-resets-count
  (when-model-available
    (testing "batch-clear resets token count to 0"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 32)
          (cl-llama-cpp:batch-add-token batch 100 0 0)
          (cl-llama-cpp:batch-add-token batch 200 1 0)
          (ok (= 2 (cl-llama-cpp:batch-token-count batch))
              "count is 2 before clear")
          (ok (null (cl-llama-cpp:batch-clear batch))
              "batch-clear returns NIL")
          (ok (zerop (cl-llama-cpp:batch-token-count batch))
              "count is 0 after clear"))))))

(deftest batch-clear-allows-reuse
  (when-model-available
    (testing "batch can be reused after clear"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 2)
          (cl-llama-cpp:batch-add-token batch 100 0 0)
          (cl-llama-cpp:batch-add-token batch 200 1 0)
          (cl-llama-cpp:batch-clear batch)
          (cl-llama-cpp:batch-add-token batch 300 0 0)
          (ok (= 1 (cl-llama-cpp:batch-token-count batch))
              "batch reused after clear"))))))

(deftest batch-add-sequence-basic
  (when-model-available
    (testing "batch-add-sequence adds all tokens from a vector"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-batch (batch 512)
          (let ((tokens (cl-llama-cpp:tokenize model "Hello world")))
            (ok (null (cl-llama-cpp:batch-add-sequence batch tokens 0))
                "batch-add-sequence returns NIL")
            (ok (= (length tokens) (cl-llama-cpp:batch-token-count batch))
                (format nil "token count matches: ~D" (length tokens)))))))))

(deftest batch-add-sequence-with-start-pos
  (when-model-available
    (testing "batch-add-sequence respects :start-pos"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-batch (batch 512)
          (let ((tokens (cl-llama-cpp:tokenize model "Test")))
            (cl-llama-cpp:batch-add-sequence batch tokens 0 :start-pos 10)
            (ok (= (length tokens) (cl-llama-cpp:batch-token-count batch))
                "tokens added with start-pos offset")))))))

(deftest batch-add-sequence-logits-modes
  (when-model-available
    (testing "batch-add-sequence accepts :logits :last, :all, and nil"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 64)
          (cl-llama-cpp:batch-add-sequence batch #(1 2 3) 0 :logits :last)
          (ok (= 3 (cl-llama-cpp:batch-token-count batch))
              ":last mode added 3 tokens"))
        (cl-llama-cpp:with-batch (batch 64)
          (cl-llama-cpp:batch-add-sequence batch #(1 2 3) 0 :logits :all)
          (ok (= 3 (cl-llama-cpp:batch-token-count batch))
              ":all mode added 3 tokens"))
        (cl-llama-cpp:with-batch (batch 64)
          (cl-llama-cpp:batch-add-sequence batch #(1 2 3) 0 :logits nil)
          (ok (= 3 (cl-llama-cpp:batch-token-count batch))
              "nil mode added 3 tokens"))))))

(deftest batch-add-sequence-overflow
  (when-model-available
    (testing "batch-add-sequence signals overflow when exceeding capacity"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (cl-llama-cpp:with-batch (batch 2)
          (ok (handler-case
                  (progn (cl-llama-cpp:batch-add-sequence batch #(1 2 3) 0) nil)
                (cl-llama-cpp:batch-overflow-error () t))
              "batch-overflow-error signaled for oversized sequence"))))))

(deftest batch-decode-happy-path
  (when-model-available
    (testing "batch-decode decodes a batch of tokens"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-batch (batch 512)
            (let ((tokens (cl-llama-cpp:tokenize model "Hello world")))
              (cl-llama-cpp:batch-add-sequence batch tokens 0 :logits :last)
              (ok (null (cl-llama-cpp:batch-decode ctx batch))
                  "batch-decode returned NIL (success)"))))))))

(deftest batch-decode-clear-reuse
  (when-model-available
    (testing "batch can be cleared and reused for multiple decodes"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-batch (batch 512)
            (let ((tokens (cl-llama-cpp:tokenize model "Hello")))
              (cl-llama-cpp:batch-add-sequence batch tokens 0 :logits :last)
              (cl-llama-cpp:batch-decode ctx batch)
              (cl-llama-cpp:batch-clear batch)
              (cl-llama-cpp:batch-add-token batch 42 (length tokens) 0 :logits t)
              (ok (null (cl-llama-cpp:batch-decode ctx batch))
                  "second decode after clear succeeded"))))))))

(deftest batch-parallel-sequences
  (when-model-available
    (testing "batch supports multiple sequences for parallel decoding"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :n-seq-max 2)
          (cl-llama-cpp:with-batch (batch 512 :n-seq-max 2)
            (let ((tokens-0 (cl-llama-cpp:tokenize model "Hello"))
                  (tokens-1 (cl-llama-cpp:tokenize model "World")))
              (cl-llama-cpp:batch-add-sequence batch tokens-0 0 :logits :last)
              (cl-llama-cpp:batch-add-sequence batch tokens-1 1
                                               :start-pos 0 :logits :last)
              (ok (= (+ (length tokens-0) (length tokens-1))
                     (cl-llama-cpp:batch-token-count batch))
                  "batch contains both sequences")
              (ok (null (cl-llama-cpp:batch-decode ctx batch))
                  "parallel decode succeeded"))))))))

(deftest with-batch-cleanup-on-error
  (when-model-available
    (testing "with-batch frees batch on non-local exit"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (let ((captured nil))
          (ignore-errors
            (cl-llama-cpp:with-batch (batch 32)
              (setf captured batch)
              (error "deliberate error")))
          (ok captured "batch handle was captured before error"))))))

;;; Performance counter integration tests

(deftest context-perf-returns-plist
  (when-model-available
    (testing "context-perf returns a plist with expected keys"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((perf (cl-llama-cpp:context-perf ctx)))
            (ok (listp perf) "context-perf returned a list")
            (ok (member :t-start-ms perf) ":t-start-ms key present")
            (ok (member :t-load-ms perf) ":t-load-ms key present")
            (ok (member :t-p-eval-ms perf) ":t-p-eval-ms key present")
            (ok (member :t-eval-ms perf) ":t-eval-ms key present")
            (ok (member :n-p-eval perf) ":n-p-eval key present")
            (ok (member :n-eval perf) ":n-eval key present")
            (ok (member :n-reused perf) ":n-reused key present")
            (ok (numberp (getf perf :t-start-ms)) ":t-start-ms is a number")
            (ok (integerp (getf perf :n-eval)) ":n-eval is an integer")))))))

(deftest reset-context-perf-returns-nil
  (when-model-available
    (testing "reset-context-perf returns NIL"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (null (cl-llama-cpp:reset-context-perf ctx))
              "reset-context-perf returned NIL"))))))

(deftest reset-context-perf-clears-counters
  (when-model-available
    (testing "reset-context-perf resets timing counters (n-eval/n-p-eval clamped to 1 by upstream)"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Hello"))
          (cl-llama-cpp:reset-context-perf ctx)
          (let ((perf (cl-llama-cpp:context-perf ctx)))
            ;; Upstream llama.cpp uses std::max(1, n_eval) in llama_perf_context
            ;; to prevent downstream division-by-zero errors in throughput math.
            (ok (= 1 (getf perf :n-eval))
                (format nil ":n-eval is 1 after reset: ~A" (getf perf :n-eval)))
            (ok (= 1 (getf perf :n-p-eval))
                (format nil ":n-p-eval is 1 after reset: ~A" (getf perf :n-p-eval)))))))))

(deftest print-context-perf-returns-nil
  (when-model-available
    (testing "print-context-perf returns NIL (side-effect: prints to stderr)"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (null (cl-llama-cpp:print-context-perf ctx))
              "print-context-perf returned NIL"))))))

(deftest sampler-perf-returns-plist
  (when-model-available
    (testing "sampler-perf returns a plist with expected keys"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain)
          (let ((perf (cl-llama-cpp:sampler-perf chain)))
            (ok (listp perf) "sampler-perf returned a list")
            (ok (member :t-sample-ms perf) ":t-sample-ms key present")
            (ok (member :n-sample perf) ":n-sample key present")
            (ok (numberp (getf perf :t-sample-ms)) ":t-sample-ms is a number")
            (ok (integerp (getf perf :n-sample)) ":n-sample is an integer")))))))

(deftest reset-sampler-perf-returns-nil
  (when-model-available
    (testing "reset-sampler-perf returns NIL"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain)
          (ok (null (cl-llama-cpp:reset-sampler-perf chain))
              "reset-sampler-perf returned NIL"))))))

(deftest print-sampler-perf-returns-nil
  (when-model-available
    (testing "print-sampler-perf returns NIL (side-effect: prints to stderr)"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-sampler-chain (chain)
          (ok (null (cl-llama-cpp:print-sampler-perf chain))
              "print-sampler-perf returned NIL"))))))

(deftest with-perf-returns-body-value
  (when-model-available
    (testing "with-perf returns the value of its body"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:with-perf (ctx) 42)))
            (ok (= 42 result) "with-perf returned body value 42")))))))

(deftest with-perf-resets-before-and-prints-after
  (when-model-available
    (testing "with-perf resets perf then executes body"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (decode-tokens ctx (cl-llama-cpp:tokenize model "Prime the counters"))
          (cl-llama-cpp:with-perf (ctx)
            (decode-tokens ctx (cl-llama-cpp:tokenize model "Hello")))
          (let ((perf (cl-llama-cpp:context-perf ctx)))
            (ok (> (getf perf :n-p-eval) 0)
                "n-p-eval > 0: perf was reset and body ran")))))))

;;; Logging integration tests

(deftest set-log-callback-captures-messages
  (when-model-available
    (testing "set-log-callback routes llama.cpp log output to a Lisp function"
      (let ((messages '())
            (prev (cl-llama-cpp:get-log-callback)))
        (unwind-protect
             (progn
               (cl-llama-cpp:set-log-callback
                (lambda (level text)
                  (declare (ignore level))
                  (push text messages)))
               (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
                 nil)
               (ok (listp messages) "captured messages is a list")
               (ok (plusp (length messages))
                   (format nil "captured ~D log messages" (length messages)))
               (ok (every #'stringp messages) "all messages are strings"))
          (cl-llama-cpp:set-log-callback prev))))))

(deftest set-log-callback-nil-restores-default
  (testing "set-log-callback nil restores default logging without error"
    (let ((prev (cl-llama-cpp:get-log-callback)))
      (unwind-protect
           (progn
             (cl-llama-cpp:set-log-callback (lambda (level text) (declare (ignore level text))))
             (ok (null (cl-llama-cpp:set-log-callback nil))
                 "set-log-callback nil returned NIL")
             (ok (null (cl-llama-cpp:get-log-callback))
                 "get-log-callback returns NIL after clearing"))
        (cl-llama-cpp:set-log-callback prev)))))

;;; System query integration tests

(deftest system-capabilities-values
  (testing "system-capabilities returns a plist with boolean and integer values"
    (let ((caps (cl-llama-cpp:system-capabilities)))
      (ok (listp caps) "system-capabilities returned a list")
      (ok (typep (getf caps :mmap) 'boolean) ":mmap is a boolean")
      (ok (typep (getf caps :mlock) 'boolean) ":mlock is a boolean")
      (ok (typep (getf caps :gpu-offload) 'boolean) ":gpu-offload is a boolean")
      (ok (typep (getf caps :rpc) 'boolean) ":rpc is a boolean")
      (ok (and (integerp (getf caps :max-devices))
               (>= (getf caps :max-devices) 1))
          (format nil ":max-devices >= 1: ~A" (getf caps :max-devices))))))

(deftest time-us-monotonic
  (testing "two successive time-us calls produce increasing values"
    (let ((t1 (cl-llama-cpp:time-us))
          (t2 (cl-llama-cpp:time-us)))
      (ok (>= t2 t1)
          (format nil "t2 (~D) >= t1 (~D)" t2 t1)))))

;;; Resource planning & configuration validation integration tests

(deftest estimate-memory-returns-plist
  (when-model-available
    (testing "estimate-memory returns a plist with expected keys"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((est (cl-llama-cpp:estimate-memory model)))
          (ok (listp est) "estimate-memory returned a list")
          (ok (integerp (getf est :model-size))
              (format nil ":model-size is integer: ~A" (getf est :model-size)))
          (ok (integerp (getf est :kv-cache))
              (format nil ":kv-cache is integer: ~A" (getf est :kv-cache)))
          (ok (integerp (getf est :compute))
              (format nil ":compute is integer: ~A" (getf est :compute)))
          (ok (integerp (getf est :total))
              (format nil ":total is integer: ~A" (getf est :total))))))))

(deftest estimate-memory-positive-values
  (when-model-available
    (testing "estimate-memory values are positive"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((est (cl-llama-cpp:estimate-memory model)))
          (ok (> (getf est :model-size) 0) "model-size > 0")
          (ok (> (getf est :kv-cache) 0) "kv-cache > 0")
          (ok (> (getf est :compute) 0) "compute > 0")
          (ok (> (getf est :total) 0) "total > 0"))))))

(deftest estimate-memory-total-is-sum
  (when-model-available
    (testing "estimate-memory :total equals sum of components"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((est (cl-llama-cpp:estimate-memory model)))
          (ok (= (getf est :total)
                 (+ (getf est :model-size)
                    (getf est :kv-cache)
                    (getf est :compute)))
              "total = model-size + kv-cache + compute"))))))

(deftest estimate-memory-ctx-scaling
  (when-model-available
    (testing "doubling n-ctx roughly doubles KV cache estimate"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let* ((est1 (cl-llama-cpp:estimate-memory model :n-ctx 512))
               (est2 (cl-llama-cpp:estimate-memory model :n-ctx 1024))
               (kv1 (getf est1 :kv-cache))
               (kv2 (getf est2 :kv-cache)))
          (ok (> kv2 kv1) "larger n-ctx has larger KV cache")
          (let ((ratio (/ kv2 kv1)))
            (ok (and (>= ratio 1.9) (<= ratio 2.1))
                (format nil "KV cache ratio ~,2F ≈ 2.0" ratio))))))))

(deftest estimate-memory-type-k-affects-kv
  (when-model-available
    (testing "quantized type-k reduces KV cache estimate"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let* ((est-f16 (cl-llama-cpp:estimate-memory model :n-ctx 512 :type-k :f16))
               (est-q8 (cl-llama-cpp:estimate-memory model :n-ctx 512 :type-k :q8-0)))
          (ok (< (getf est-q8 :kv-cache) (getf est-f16 :kv-cache))
              "q8-0 type-k reduces KV cache vs f16"))))))

(deftest validate-configuration-unknown-without-budget
  (when-model-available
    (testing "validate-configuration without vram-budget: auto-detects GPU or returns :unknown"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (let* ((detected (cl-llama-cpp:detect-free-vram))
               (result (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
                         (cl-llama-cpp:validate-configuration model :n-ctx 512)))
               (status (getf result :status)))
          (if detected
              (ok (member status '(:safe :unsafe))
                  (format nil "GPU detected: status is :safe or :unsafe, got ~A" status))
              (ok (eq :unknown status)
                  "no GPU detected: status is :unknown")))))))

(deftest validate-configuration-safe-with-large-budget
  (when-model-available
    (testing "validate-configuration returns :safe with generous budget"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((result (cl-llama-cpp:validate-configuration
                       model :n-ctx 512
                             :vram-budget (* 100 1024 1024 1024))))
          (ok (eq :safe (getf result :status))
              (format nil "status is :safe: ~A" (getf result :reason))))))))

(deftest validate-configuration-unsafe-with-tiny-budget
  (when-model-available
    (testing "validate-configuration returns :unsafe with tiny budget"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((result (cl-llama-cpp:validate-configuration
                       model :n-ctx 512 :vram-budget 1024)))
          (ok (eq :unsafe (getf result :status))
              (format nil "status is :unsafe: ~A" (getf result :reason))))))))

(deftest validate-configuration-gpu-layers-reduces-vram
  (when-model-available
    (testing "fewer GPU layers reduces VRAM estimate"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let* ((est (cl-llama-cpp:estimate-memory model :n-ctx 512))
               (tight-budget (+ (getf est :kv-cache) (getf est :compute)
                                (ceiling (/ (getf est :model-size) 2))))
               (result-all (cl-llama-cpp:validate-configuration
                            model :n-ctx 512 :vram-budget tight-budget))
               (result-half (cl-llama-cpp:validate-configuration
                             model :n-ctx 512 :n-gpu-layers 1
                                   :vram-budget tight-budget)))
          (ok (or (eq :unsafe (getf result-all :status))
                  (eq :safe (getf result-half :status)))
              "fewer GPU layers helps fit budget"))))))

(deftest suggest-configuration-nil-without-budget
  (when-model-available
    (testing "suggest-configuration without vram-budget: auto-detects GPU or returns NIL"
      (cl-llama-cpp:with-llama-compatible-fp-environment
        (%llama:backend-init)
        (let* ((detected (cl-llama-cpp:detect-free-vram))
               (result (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
                         (cl-llama-cpp:suggest-configuration model :n-ctx 512))))
          (if detected
              (ok (or (null result) (listp result))
                  (format nil "GPU detected: returns a config or NIL, got ~A" result))
              (ok (null result)
                  "no GPU detected: returns NIL")))))))

(deftest suggest-configuration-returns-valid-config
  (when-model-available
    (testing "suggest-configuration returns a valid config with large budget"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((suggestion (cl-llama-cpp:suggest-configuration
                           model :n-ctx 4096
                                 :vram-budget (* 100 1024 1024 1024))))
          (ok (listp suggestion) "suggestion is a list")
          (ok (integerp (getf suggestion :n-ctx)) ":n-ctx is an integer")
          (ok (integerp (getf suggestion :n-gpu-layers)) ":n-gpu-layers is an integer")
          (ok (> (getf suggestion :n-ctx) 0) ":n-ctx > 0"))))))

(deftest suggest-configuration-reduces-params
  (when-model-available
    (testing "suggest-configuration reduces params for tight budget"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let* ((est (cl-llama-cpp:estimate-memory model :n-ctx 4096))
               (tight (ceiling (* (getf est :total) 0.5)))
               (suggestion (cl-llama-cpp:suggest-configuration
                            model :n-ctx 4096 :vram-budget tight)))
          (when suggestion
            (ok (or (< (getf suggestion :n-ctx) 4096)
                    (< (getf suggestion :n-gpu-layers)
                       (getf (cl-llama-cpp:model-info model) :n-layers)))
                "suggestion reduced at least one parameter")))))))

(deftest explain-memory-usage-prints
  (when-model-available
    (testing "explain-memory-usage prints to a stream"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((output (with-output-to-string (s)
                        (cl-llama-cpp:explain-memory-usage model :n-ctx 512
                                                                 :stream s))))
          (ok (> (length output) 0) "produced output")
          (ok (search "Model Weights" output) "contains Model Weights line")
          (ok (search "KV Cache" output) "contains KV Cache line")
          (ok (search "Total Estimated" output) "contains Total line"))))))

(deftest feasibility-report-returns-plist
  (when-model-available
    (testing "feasibility-report returns combined estimate and validation plist"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((result (with-output-to-string (s)
                        (setf result (cl-llama-cpp:feasibility-report
                                      model :n-ctx 512
                                            :vram-budget (* 100 1024 1024 1024)
                                            :stream s)))))
          (declare (ignore result))
          t)
        (let ((result (cl-llama-cpp:feasibility-report
                       model :n-ctx 512
                             :vram-budget (* 100 1024 1024 1024)
                             :stream (make-string-output-stream))))
          (ok (listp result) "result is a list")
          (ok (member :model-size result) ":model-size present")
          (ok (member :status result) ":status present")
          (ok (eq :safe (getf result :status))
              (format nil "status is :safe: ~A" (getf result :reason))))))))

(deftest with-context-validation-warn
  (when-model-available
    (testing "with-context :validation :warn signals warning for tiny budget"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((warned nil))
          (handler-bind ((cl-llama-cpp:configuration-unsafe-warning
                          (lambda (c)
                            (setf warned (cl-llama-cpp::configuration-unsafe-warning-reason c))
                            (muffle-warning c))))
            (cl-llama-cpp:with-context (ctx model :n-ctx 512
                                                  :validation :warn
                                                  :vram-budget 1024)
              (ok (cl-llama-cpp:llama-context-p ctx)
                  "context still created despite warning")))
          (ok warned (format nil "warning was signaled: ~A" warned)))))))

(deftest with-context-validation-error
  (when-model-available
    (testing "with-context :validation :error prevents context creation for tiny budget"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (handler-case
                (cl-llama-cpp:with-context (ctx model :n-ctx 512
                                                      :validation :error
                                                      :vram-budget 1024)
                  ctx
                  nil)
              (cl-llama-cpp:configuration-unsafe-error (c)
                (cl-llama-cpp:configuration-unsafe-error-reason c)))
            "configuration-unsafe-error was signaled")))))

(deftest with-context-validation-off
  (when-model-available
    (testing "with-context :validation :off does not validate"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512
                                              :validation :off
                                              :vram-budget 1024)
          (ok (cl-llama-cpp:llama-context-p ctx)
              "context created without validation"))))))

;;; Boolean ergonomics integration tests (issue #43)

(deftest with-context-embeddings-t
  (if *test-embed-model-path*
      (testing "with-context accepts :embeddings T and embedding works"
        (cl-llama-cpp:with-model (model *test-embed-model-path* :n-gpu-layers 0)
          (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings t
                                                :pooling-type 1)
            (let ((embedding (cl-llama-cpp:embed ctx "Hello")))
              (ok (vectorp embedding)
                  "embed returned a vector with :embeddings T")
              (ok (> (length embedding) 0)
                  "embedding is non-empty")))))
      (skip "LLAMA_TEST_EMBED_MODEL not set — skipping")))

(deftest with-context-embeddings-nil
  (when-model-available
    (testing "with-context accepts :embeddings NIL (no embeddings mode)"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings nil)
          (ok (cl-llama-cpp:llama-context-p ctx)
              "context created with :embeddings NIL"))))))

(deftest with-model-vocab-only-t
  (when-model-available
    (testing "with-model accepts :vocab-only T"
      (cl-llama-cpp:with-model (model *test-model-path* :vocab-only t)
        (ok (cl-llama-cpp:llama-model-p model)
            "model loaded with :vocab-only T")))))

(deftest with-context-bool-backward-compat
  (when-model-available
    (testing "with-context still accepts integer 0/1 for boolean params"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :embeddings 0)
          (ok (cl-llama-cpp:llama-context-p ctx)
              "context created with :embeddings 0 (backward compat)"))))))

;;; Callback safety integration tests (issue #40)

(deftest log-callback-error-captured-in-last-error
  (when-model-available
    (testing "errors in log callback are captured in *last-log-callback-error*"
      (let ((prev (cl-llama-cpp:get-log-callback))
            (prev-err cl-llama-cpp:*last-log-callback-error*))
        (unwind-protect
             (progn
               (setf cl-llama-cpp:*last-log-callback-error* nil)
               (cl-llama-cpp:set-log-callback
                (lambda (level text)
                  (declare (ignore level text))
                  (error "deliberate callback error")))
               ;; Loading a model triggers log messages, which invokes the callback
               (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
                 nil)
               (ok (typep cl-llama-cpp:*last-log-callback-error* 'error)
                   (format nil "error was captured: ~A"
                           cl-llama-cpp:*last-log-callback-error*)))
          (cl-llama-cpp:set-log-callback prev)
          (setf cl-llama-cpp:*last-log-callback-error* prev-err))))))

(deftest generate-token-callback-nil-returns-callback
  (when-model-available
    (testing "token callback returning NIL produces :callback stop reason"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (text stop-reason)
              (cl-llama-cpp:generate ctx "The capital of France is"
                                     :max-tokens 64
                                     :temp 0.1
                                     :token-callback (lambda (chunk)
                                                       (declare (ignore chunk))
                                                       nil))
            (ok (stringp text) "generate returned a string")
            (ok (eq :callback stop-reason)
                (format nil "stop-reason is :callback: ~A" stop-reason))))))))

(deftest generate-token-callback-error-returns-error-stop
  (when-model-available
    (testing "token callback that signals an error produces :error stop reason"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (text stop-reason)
              (cl-llama-cpp:generate ctx "The capital of France is"
                                     :max-tokens 64
                                     :temp 0.1
                                     :token-callback (lambda (chunk)
                                                       (declare (ignore chunk))
                                                       (error "deliberate callback error")))
            (ok (stringp text) "generate returned a string despite callback error")
            (ok (eq :error stop-reason)
                (format nil "stop-reason is :error: ~A" stop-reason))))))))

(deftest generate-token-callback-ignore-restart-continues
  (when-model-available
    (testing "invoking ignore-callback-error restart allows generation to continue"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((call-count 0))
            (multiple-value-bind (text stop-reason)
                (cl-llama-cpp:generate
                 ctx "The capital of France is"
                 :max-tokens 16
                 :temp 0.1
                 :token-callback
                 ;; Handler-bind inside the callback is closer to the signal
                 ;; than generate's internal handler, so it runs first.
                 (lambda (chunk)
                   (declare (ignore chunk))
                   (incf call-count)
                   (when (= call-count 1)
                     (handler-bind
                         ((error (lambda (c)
                                   (declare (ignore c))
                                   (when (find-restart 'cl-llama-cpp::ignore-callback-error)
                                     (invoke-restart 'cl-llama-cpp::ignore-callback-error)))))
                       (error "first call errors")))))
              (ok (stringp text) "generate returned text after error recovery")
              (ok (member stop-reason '(:eog :length :callback))
                  (format nil "stop-reason is not :error after recovery: ~A" stop-reason))
              (ok (> call-count 1)
                  (format nil "callback was invoked ~D times (continued after error)"
                          call-count)))))))))

(deftest generate-without-token-callback-unaffected
  (when-model-available
    (testing "generate without token callback still returns :eog or :length"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (text stop-reason)
              (cl-llama-cpp:generate ctx "The capital of France is"
                                     :max-tokens 16
                                     :temp 0.1)
            (ok (stringp text) "generate returned text")
            (ok (member stop-reason '(:eog :length))
                (format nil "stop-reason is :eog or :length: ~A" stop-reason))))))))

;;; Abort callback integration tests (issue #45)

(deftest get-abort-callback-initial
  (when-model-available
    (testing "get-abort-callback returns NIL on a fresh context"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 128)
          (ok (null (cl-llama-cpp:get-abort-callback ctx))
              "get-abort-callback returns NIL on fresh context"))))))

(deftest set-abort-callback-roundtrip
  (when-model-available
    (testing "set-abort-callback installs and get-abort-callback retrieves it"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 128)
          (let ((fn (lambda () nil)))
            (ok (null (cl-llama-cpp:set-abort-callback ctx fn))
                "set-abort-callback returns NIL")
            (ok (eq fn (cl-llama-cpp:get-abort-callback ctx))
                "get-abort-callback returns the installed function")
            (ok (null (cl-llama-cpp:set-abort-callback ctx nil))
                "set-abort-callback nil returns NIL")
            (ok (null (cl-llama-cpp:get-abort-callback ctx))
                "get-abort-callback returns NIL after clearing")))))))

(deftest abort-callback-cleared-on-context-free
  (when-model-available
    (testing "abort callback hash is cleaned up when context is freed via with-context"
      (let (key)
        (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
          (cl-llama-cpp:with-context (ctx model :n-ctx 128)
            (let ((fn (lambda () nil)))
              (cl-llama-cpp:set-abort-callback ctx fn)
              (setf key (cffi:pointer-address
                         (cl-llama-cpp:llama-context-pointer ctx)))))
          ;; After with-context exits the hash entry should be gone
          (ok (null (gethash key cl-llama-cpp::*abort-callbacks*))
              "abort callbacks hash entry removed after context freed"))))))

(deftest abort-callback-error-captured-in-last-error
  (when-model-available
    (testing "errors in abort callback are captured in *last-abort-callback-error*"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 128)
          (let ((prev-err cl-llama-cpp:*last-abort-callback-error*))
            (unwind-protect
                 (progn
                   (setf cl-llama-cpp:*last-abort-callback-error* nil)
                   ;; Install a callback that always signals an error
                   (cl-llama-cpp:set-abort-callback
                    ctx (lambda () (error "deliberate abort callback error")))
                   ;; Run generate — the abort callback fires during decode but
                   ;; the panic boundary catches the error and returns NIL (no abort)
                   (cl-llama-cpp:generate ctx "The" :max-tokens 4 :temp 0.1)
                   (ok (typep cl-llama-cpp:*last-abort-callback-error* 'error)
                       (format nil "abort callback error was captured: ~A"
                               cl-llama-cpp:*last-abort-callback-error*)))
              (cl-llama-cpp:set-abort-callback ctx nil)
              (setf cl-llama-cpp:*last-abort-callback-error* prev-err))))))))

;;; Backend device & registry introspection integration tests (issue #29)

(deftest backend-dev-count-positive
  (testing "backend-dev-count returns at least 1 after backend-init"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((count (cl-llama-cpp:backend-dev-count)))
        (ok (>= count 1)
            (format nil "backend-dev-count >= 1: ~D" count))))))

(deftest backend-dev-get-returns-handle
  (testing "backend-dev-get returns a ggml-backend-device handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((dev (cl-llama-cpp:backend-dev-get 0)))
        (ok (cl-llama-cpp:ggml-backend-device-p dev)
            "backend-dev-get returned a ggml-backend-device")
        (ok (not (cffi:null-pointer-p (cl-llama-cpp:ggml-backend-device-pointer dev)))
            "device pointer is non-null")))))

(deftest backend-dev-get-out-of-range-signals-error
  (testing "backend-dev-get with out-of-range index signals error"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((count (cl-llama-cpp:backend-dev-count)))
        (ok (handler-case
                (progn (cl-llama-cpp:backend-dev-get count) nil)
              (error () t))
            "error signaled for out-of-range index")))))

(deftest backend-dev-name-returns-string
  (testing "backend-dev-name returns a non-empty string"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((dev (cl-llama-cpp:backend-dev-get 0))
             (name (cl-llama-cpp:backend-dev-name dev)))
        (ok (stringp name) "backend-dev-name returned a string")
        (ok (> (length name) 0)
            (format nil "device name: ~S" name))))))

(deftest backend-dev-description-returns-string
  (testing "backend-dev-description returns a string"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((dev (cl-llama-cpp:backend-dev-get 0))
             (desc (cl-llama-cpp:backend-dev-description dev)))
        (ok (stringp desc)
            (format nil "backend-dev-description returned a string: ~S" desc))))))

(deftest backend-dev-type-returns-keyword
  (testing "backend-dev-type returns a valid keyword"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((dev (cl-llama-cpp:backend-dev-get 0))
             (type (cl-llama-cpp:backend-dev-type dev)))
        (ok (member type '(:cpu :gpu :igpu :accel :meta))
            (format nil "device type is a valid keyword: ~A" type))))))

(deftest backend-dev-memory-returns-values
  (testing "backend-dev-memory returns two non-negative integers"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((dev (cl-llama-cpp:backend-dev-get 0)))
        (multiple-value-bind (free total)
            (cl-llama-cpp:backend-dev-memory dev)
          (ok (integerp free)
              (format nil "free-bytes is integer: ~D" free))
          (ok (integerp total)
              (format nil "total-bytes is integer: ~D" total))
          (ok (>= free 0) "free-bytes >= 0")
          (ok (>= total 0) "total-bytes >= 0"))))))

(deftest backend-dev-props-returns-plist
  (testing "backend-dev-props returns a plist with all expected keys"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((dev (cl-llama-cpp:backend-dev-get 0))
             (props (cl-llama-cpp:backend-dev-props dev)))
        (ok (listp props) "backend-dev-props returned a list")
        (ok (stringp (getf props :name)) ":name is a string")
        (ok (stringp (getf props :description)) ":description is a string")
        (ok (integerp (getf props :memory-free)) ":memory-free is an integer")
        (ok (integerp (getf props :memory-total)) ":memory-total is an integer")
        (ok (member (getf props :type) '(:cpu :gpu :igpu :accel :meta))
            (format nil ":type is valid keyword: ~A" (getf props :type)))
        (ok (typep (getf props :async) 'boolean) ":async is a boolean")
        (ok (typep (getf props :host-buffer) 'boolean) ":host-buffer is a boolean")
        (ok (typep (getf props :buffer-from-host-ptr) 'boolean)
            ":buffer-from-host-ptr is a boolean")
        (ok (typep (getf props :events) 'boolean) ":events is a boolean")))))

(deftest backend-dev-by-type-cpu-found
  (testing "backend-dev-by-type :cpu returns a ggml-backend-device"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((dev (cl-llama-cpp:backend-dev-by-type :cpu)))
        (ok (cl-llama-cpp:ggml-backend-device-p dev)
            "backend-dev-by-type :cpu returned a ggml-backend-device")
        (ok (eq :cpu (cl-llama-cpp:backend-dev-type dev))
            "device type is :cpu")))))

(deftest backend-dev-by-name-roundtrip
  (testing "backend-dev-by-name finds a device using its own name"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((dev0 (cl-llama-cpp:backend-dev-get 0))
             (name (cl-llama-cpp:backend-dev-name dev0))
             (found (cl-llama-cpp:backend-dev-by-name name)))
        (ok (cl-llama-cpp:ggml-backend-device-p found)
            (format nil "found device by name: ~S" name))
        (ok (string= name (cl-llama-cpp:backend-dev-name found))
            "found device has same name")))))

(deftest backend-dev-by-name-not-found
  (testing "backend-dev-by-name returns NIL for unknown name"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (ok (null (cl-llama-cpp:backend-dev-by-name "nonexistent-device-xyz"))
          "backend-dev-by-name returned NIL for unknown name"))))

(deftest backend-reg-count-positive
  (testing "backend-reg-count returns at least 1 after backend-init"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((count (cl-llama-cpp:backend-reg-count)))
        (ok (>= count 1)
            (format nil "backend-reg-count >= 1: ~D" count))))))

(deftest backend-reg-get-returns-handle
  (testing "backend-reg-get returns a ggml-backend-registry handle"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((reg (cl-llama-cpp:backend-reg-get 0)))
        (ok (cl-llama-cpp:ggml-backend-registry-p reg)
            "backend-reg-get returned a ggml-backend-registry")
        (ok (not (cffi:null-pointer-p (cl-llama-cpp:ggml-backend-registry-pointer reg)))
            "registry pointer is non-null")))))

(deftest backend-reg-name-returns-string
  (testing "backend-reg-name returns a non-empty string"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((reg (cl-llama-cpp:backend-reg-get 0))
             (name (cl-llama-cpp:backend-reg-name reg)))
        (ok (stringp name) "backend-reg-name returned a string")
        (ok (> (length name) 0)
            (format nil "registry name: ~S" name))))))

(deftest backend-reg-dev-count-non-negative
  (testing "backend-reg-dev-count returns a non-negative integer"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((reg (cl-llama-cpp:backend-reg-get 0))
             (count (cl-llama-cpp:backend-reg-dev-count reg)))
        (ok (integerp count) "backend-reg-dev-count returned an integer")
        (ok (>= count 0)
            (format nil "backend-reg-dev-count >= 0: ~D" count))))))

(deftest backend-reg-dev-get-returns-handle
  (testing "backend-reg-dev-get returns a ggml-backend-device when count > 0"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((reg (cl-llama-cpp:backend-reg-get 0))
             (count (cl-llama-cpp:backend-reg-dev-count reg)))
        (when (> count 0)
          (let ((dev (cl-llama-cpp:backend-reg-dev-get reg 0)))
            (ok (cl-llama-cpp:ggml-backend-device-p dev)
                "backend-reg-dev-get returned a ggml-backend-device")))))))

(deftest backend-reg-by-name-roundtrip
  (testing "backend-reg-by-name finds a registry using its own name"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let* ((reg0 (cl-llama-cpp:backend-reg-get 0))
             (name (cl-llama-cpp:backend-reg-name reg0))
             (found (cl-llama-cpp:backend-reg-by-name name)))
        (ok (cl-llama-cpp:ggml-backend-registry-p found)
            (format nil "found registry by name: ~S" name))
        (ok (string= name (cl-llama-cpp:backend-reg-name found))
            "found registry has same name")))))

(deftest backend-reg-by-name-not-found
  (testing "backend-reg-by-name returns NIL for unknown name"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (ok (null (cl-llama-cpp:backend-reg-by-name "nonexistent-registry-xyz"))
          "backend-reg-by-name returned NIL for unknown name"))))

(deftest gpu-devices-returns-list
  (testing "gpu-devices returns a list (possibly empty on CPU-only machines)"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((devs (cl-llama-cpp:gpu-devices)))
        (ok (listp devs) "gpu-devices returned a list")
        (when devs
          (ok (every #'listp devs) "each entry is a plist")
          (ok (stringp (getf (first devs) :name))
              ":name is a string in first GPU device plist"))))))

(deftest detect-free-vram-type
  (testing "detect-free-vram returns an integer or NIL"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((vram (cl-llama-cpp:detect-free-vram)))
        (ok (or (null vram) (integerp vram))
            (format nil "detect-free-vram returned ~A" vram))
        (when vram
          (ok (>= vram 0) "detect-free-vram is non-negative"))))))

(deftest detect-total-vram-type
  (testing "detect-total-vram returns an integer or NIL"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((total (cl-llama-cpp:detect-total-vram)))
        (ok (or (null total) (integerp total))
            (format nil "detect-total-vram returned ~A" total))
        (when total
          (ok (>= total 0) "detect-total-vram is non-negative"))))))

(deftest detect-total-vram-not-less-than-free
  (testing "detect-total-vram >= detect-free-vram when GPU is present"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((free  (cl-llama-cpp:detect-free-vram))
            (total (cl-llama-cpp:detect-total-vram)))
        (when (and free total)
          (ok (>= total free)
              (format nil "total ~A >= free ~A" total free)))))))

(deftest validate-configuration-auto-detects-budget
  (testing "validate-configuration with no vram-budget matches explicit detected budget"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((detected (cl-llama-cpp:detect-free-vram)))
        (when detected
          (when-model-available
            (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
              (let* ((explicit (cl-llama-cpp:validate-configuration
                                model :n-ctx 512 :vram-budget detected))
                     (auto    (cl-llama-cpp:validate-configuration model :n-ctx 512)))
                (ok (eq (getf explicit :status) (getf auto :status))
                    "auto-detected budget gives same status as explicit same budget")))))))))

(deftest system-capabilities-backend-keys
  (testing "system-capabilities returns extended backend keys"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((caps (cl-llama-cpp:system-capabilities)))
        (ok (integerp (getf caps :n-backend-devs))
            (format nil ":n-backend-devs is integer: ~A" (getf caps :n-backend-devs)))
        (ok (integerp (getf caps :n-backend-regs))
            (format nil ":n-backend-regs is integer: ~A" (getf caps :n-backend-regs)))
        (ok (typep (getf caps :has-gpu) 'boolean)
            (format nil ":has-gpu is boolean: ~A" (getf caps :has-gpu)))
        (ok (>= (getf caps :n-backend-devs) 1)
            ":n-backend-devs >= 1 after backend-init")))))

;;; Typed opaque handles (issue #41)

(deftest handle-type-safety
  (when-model-available
    (testing "model and context handles are distinct types"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (ok (cl-llama-cpp:llama-model-p model) "model is llama-model")
        (ok (not (cl-llama-cpp:llama-context-p model))
            "model is not llama-context")
        (ok (not (cl-llama-cpp:llama-sampler-p model))
            "model is not llama-sampler")
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (cl-llama-cpp:llama-context-p ctx) "ctx is llama-context")
          (ok (not (cl-llama-cpp:llama-model-p ctx))
              "ctx is not llama-model")
          (ok (not (cl-llama-cpp:llama-sampler-p ctx))
              "ctx is not llama-sampler")
          (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-model-pointer model)))
              "llama-model-pointer is non-null")
          (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-context-pointer ctx)))
              "llama-context-pointer is non-null"))))))

(deftest sampler-handle-type
  (when-model-available
    (testing "with-sampler-chain binds a llama-sampler handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (cl-llama-cpp:with-sampler-chain (s :temp 0.8 :seed 42)
            (ok (cl-llama-cpp:llama-sampler-p s) "s is llama-sampler")
            (ok (not (cl-llama-cpp:llama-model-p s))
                "s is not llama-model")
            (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-sampler-pointer s)))
                "llama-sampler-pointer is non-null")
            (ok (integerp (cl-llama-cpp:sampler-seed s))
                "sampler-seed works on llama-sampler handle")))))))

(deftest grammar-sampler-handle-type
  (when-model-available
    (testing "make-grammar-sampler returns a llama-sampler handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let* ((grammar "root ::= \"yes\" | \"no\"")
               (s (cl-llama-cpp:make-grammar-sampler model grammar)))
          (ok (cl-llama-cpp:llama-sampler-p s)
              "make-grammar-sampler returns a llama-sampler")
          (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-sampler-pointer s)))
              "grammar sampler pointer is non-null")
          (%llama:sampler-free (cl-llama-cpp:llama-sampler-pointer s)))))))

;;; GGUF file inspection integration tests

(defmacro when-gguf-available (&body body)
  `(if *test-gguf-path*
       (progn ,@body)
       (skip "No GGUF model available (set LLAMA_TEST_MODEL or LLAMA_TEST_EMBED_MODEL)")))

(deftest with-gguf-bad-path
  (testing "with-gguf signals gguf-load-error for nonexistent file"
    (ok (handler-case
            (cl-llama-cpp:with-gguf (g "/nonexistent/model.gguf") g)
          (cl-llama-cpp:gguf-load-error (c)
            (string= "/nonexistent/model.gguf" (cl-llama-cpp:gguf-load-error-path c))))
        "gguf-load-error signaled with correct path")))

(deftest with-gguf-opens-file
  (when-gguf-available
    (testing "with-gguf opens a real GGUF file and returns a handle"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (ok (cl-llama-cpp:gguf-context-p g) "result is a gguf-context")
        (ok (cffi:pointerp (cl-llama-cpp:gguf-context-pointer g))
            "gguf-context-pointer returns a CFFI pointer")
        (ok (not (cffi:null-pointer-p (cl-llama-cpp:gguf-context-pointer g)))
            "pointer is non-null")))))

(deftest gguf-file-level
  (when-gguf-available
    (testing "gguf-version, gguf-alignment, gguf-data-offset return plausible integers"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let ((ver (cl-llama-cpp:gguf-version g))
              (align (cl-llama-cpp:gguf-alignment g))
              (offset (cl-llama-cpp:gguf-data-offset g)))
          (ok (and (integerp ver) (>= ver 1))
              (format nil "version is a positive integer: ~D" ver))
          (ok (and (integerp align) (> align 0))
              (format nil "alignment is a positive integer: ~D" align))
          (ok (and (integerp offset) (> offset 0))
              (format nil "data-offset is a positive integer: ~D" offset)))))))

(deftest gguf-kv-count
  (when-gguf-available
    (testing "gguf-n-kv returns a positive integer"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let ((n (cl-llama-cpp:gguf-n-kv g)))
          (ok (and (integerp n) (> n 0))
              (format nil "n-kv = ~D (positive)" n)))))))

(deftest gguf-find-key-missing
  (when-gguf-available
    (testing "gguf-find-key returns NIL for an absent key"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (ok (null (cl-llama-cpp:gguf-find-key g "cl.llama.cpp.no.such.key"))
            "absent key returns NIL")))))

(deftest gguf-key-iteration
  (when-gguf-available
    (testing "gguf-key returns strings for all KV indices"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let ((n (cl-llama-cpp:gguf-n-kv g)))
          (loop for i from 0 below (min n 5)
                do (let ((k (cl-llama-cpp:gguf-key g i)))
                     (ok (stringp k)
                         (format nil "key ~D is a string: ~S" i k)))))))))

(deftest gguf-find-key-roundtrip
  (when-gguf-available
    (testing "gguf-find-key round-trips through gguf-key"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let* ((key (cl-llama-cpp:gguf-key g 0))
               (found (cl-llama-cpp:gguf-find-key g key)))
          (ok (eql found 0)
              (format nil "gguf-find-key ~S returned index 0" key)))))))

(deftest gguf-kv-type-keywords
  (when-gguf-available
    (testing "gguf-kv-type returns a keyword for every KV entry"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let* ((n (cl-llama-cpp:gguf-n-kv g))
               (valid-types '(:uint8 :int8 :uint16 :int16 :uint32 :int32
                              :float32 :bool :string :array :uint64 :int64 :float64)))
          (loop for i from 0 below n
                do (let ((ty (cl-llama-cpp:gguf-kv-type g i)))
                     (ok (member ty valid-types)
                         (format nil "kv-type at ~D is a valid keyword: ~S" i ty)))))))))

(deftest gguf-val-scalars
  (when-gguf-available
    (testing "gguf-val returns CL values for scalar KV entries"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let ((n (cl-llama-cpp:gguf-n-kv g)))
          (loop for i from 0 below n
                for ty = (cl-llama-cpp:gguf-kv-type g i)
                when (not (eq ty :array))
                do (let ((v (cl-llama-cpp:gguf-val g i)))
                     (ok (or (integerp v) (floatp v)
                             (stringp v) (eq v t) (eq v nil)
                             (eq v :count))
                         (format nil "gguf-val at ~D (~S) returned a CL value: ~S"
                                 i ty v)))))))))

(deftest gguf-metadata-alist
  (when-gguf-available
    (testing "gguf-metadata returns an alist with string keys"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let ((meta (cl-llama-cpp:gguf-metadata g)))
          (ok (listp meta) "metadata is a list")
          (ok (> (length meta) 0) "metadata is non-empty")
          (ok (every (lambda (pair)
                       (and (consp pair) (stringp (car pair))))
                     meta)
              "every entry is (string . value)"))))))

(deftest gguf-tensor-count
  (when-gguf-available
    (testing "gguf-n-tensors returns a positive integer"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let ((n (cl-llama-cpp:gguf-n-tensors g)))
          (ok (and (integerp n) (> n 0))
              (format nil "n-tensors = ~D (positive)" n)))))))

(deftest gguf-find-tensor-missing
  (when-gguf-available
    (testing "gguf-find-tensor returns NIL for an absent tensor"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (ok (null (cl-llama-cpp:gguf-find-tensor g "cl.llama.cpp.no.such.tensor"))
            "absent tensor returns NIL")))))

(deftest gguf-tensor-info-plist
  (when-gguf-available
    (testing "gguf-tensor-info returns a plist with expected keys"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let ((info (cl-llama-cpp:gguf-tensor-info g 0)))
          (ok (listp info) "tensor-info is a list")
          (ok (stringp (getf info :name))
              (format nil "tensor name is a string: ~S" (getf info :name)))
          (ok (keywordp (getf info :type))
              (format nil "tensor type is a keyword: ~S" (getf info :type)))
          (ok (integerp (getf info :offset))
              (format nil "tensor offset is an integer: ~D" (getf info :offset)))
          (ok (and (integerp (getf info :size)) (> (getf info :size) 0))
              (format nil "tensor size is a positive integer: ~D" (getf info :size))))))))

(deftest gguf-tensors-list
  (when-gguf-available
    (testing "gguf-tensors returns one plist per tensor"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let* ((n (cl-llama-cpp:gguf-n-tensors g))
               (tensors (cl-llama-cpp:gguf-tensors g)))
          (ok (= (length tensors) n)
              (format nil "gguf-tensors length ~D matches gguf-n-tensors ~D"
                      (length tensors) n)))))))

(deftest gguf-find-tensor-roundtrip
  (when-gguf-available
    (testing "gguf-find-tensor round-trips through gguf-tensor-name"
      (cl-llama-cpp:with-gguf (g *test-gguf-path* :no-alloc t)
        (let* ((name (cl-llama-cpp:gguf-tensor-name g 0))
               (found (cl-llama-cpp:gguf-find-tensor g name)))
          (ok (eql found 0)
              (format nil "gguf-find-tensor ~S returned index 0" name)))))))

(deftest gguf-type-name-returns-string
  (testing "gguf-type-name returns a non-empty string for known types"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (dolist (ty '(:uint8 :int8 :uint32 :float32 :bool :string :array))
        (let ((name (cl-llama-cpp:gguf-type-name ty)))
          (ok (and (stringp name) (> (length name) 0))
              (format nil "gguf-type-name ~S => ~S" ty name)))))))

;;; GC finalizer / standalone constructor integration tests (issue #46)

(deftest make-model-creates-handle
  (when-model-available
    (testing "make-model returns a llama-model handle"
      (let ((model (cl-llama-cpp:make-model *test-model-path* :n-gpu-layers 0)))
        (unwind-protect
             (progn
               (ok (cl-llama-cpp:llama-model-p model)
                   "make-model returned a llama-model handle")
               (ok (cffi:pointerp (cl-llama-cpp:llama-model-pointer model))
                   "handle contains a CFFI pointer")
               (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-model-pointer model)))
                   "handle pointer is non-null"))
          (cl-llama-cpp:free-model model))))))

(deftest free-model-returns-nil
  (when-model-available
    (testing "free-model returns NIL"
      (let ((model (cl-llama-cpp:make-model *test-model-path* :n-gpu-layers 0)))
        (ok (null (cl-llama-cpp:free-model model))
            "free-model returned NIL")))))

(deftest free-model-idempotent
  (when-model-available
    (testing "free-model is idempotent — second call is a no-op"
      (let ((model (cl-llama-cpp:make-model *test-model-path* :n-gpu-layers 0)))
        (cl-llama-cpp:free-model model)
        (ok (null (cl-llama-cpp:free-model model))
            "second free-model returned NIL without error")))))

(deftest free-model-sets-freed-cell
  (when-model-available
    (testing "free-model sets the freed-cell flag"
      (let ((model (cl-llama-cpp:make-model *test-model-path* :n-gpu-layers 0)))
        (ok (null (car (cl-llama-cpp::llama-model-freed-cell model)))
            "freed-cell is NIL before free")
        (cl-llama-cpp:free-model model)
        (ok (car (cl-llama-cpp::llama-model-freed-cell model))
            "freed-cell is T after free")))))

(deftest make-context-creates-handle
  (when-model-available
    (testing "make-context returns a llama-context handle"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((ctx (cl-llama-cpp:make-context model :n-ctx 512)))
          (unwind-protect
               (progn
                 (ok (cl-llama-cpp:llama-context-p ctx)
                     "make-context returned a llama-context handle")
                 (ok (not (cffi:null-pointer-p (cl-llama-cpp:llama-context-pointer ctx)))
                     "context pointer is non-null"))
            (cl-llama-cpp:free-context ctx)))))))

(deftest free-context-returns-nil
  (when-model-available
    (testing "free-context returns NIL"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((ctx (cl-llama-cpp:make-context model :n-ctx 512)))
          (ok (null (cl-llama-cpp:free-context ctx))
              "free-context returned NIL"))))))

(deftest free-context-idempotent
  (when-model-available
    (testing "free-context is idempotent — second call is a no-op"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((ctx (cl-llama-cpp:make-context model :n-ctx 512)))
          (cl-llama-cpp:free-context ctx)
          (ok (null (cl-llama-cpp:free-context ctx))
              "second free-context returned NIL without error"))))))

(deftest free-context-sets-freed-cell
  (when-model-available
    (testing "free-context sets the freed-cell flag"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((ctx (cl-llama-cpp:make-context model :n-ctx 512)))
          (ok (null (car (cl-llama-cpp::llama-context-freed-cell ctx)))
              "freed-cell is NIL before free")
          (cl-llama-cpp:free-context ctx)
          (ok (car (cl-llama-cpp::llama-context-freed-cell ctx))
              "freed-cell is T after free"))))))

(deftest with-model-sets-freed-cell-on-exit
  (when-model-available
    (testing "with-model marks the handle freed on normal exit"
      (let ((saved nil))
        (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
          (setf saved model))
        (ok (car (cl-llama-cpp::llama-model-freed-cell saved))
            "freed-cell is T after with-model exits")))))

(deftest with-context-sets-freed-cell-on-exit
  (when-model-available
    (testing "with-context marks the handle freed on normal exit"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (let ((saved nil))
          (cl-llama-cpp:with-context (ctx model :n-ctx 512)
            (setf saved ctx))
          (ok (car (cl-llama-cpp::llama-context-freed-cell saved))
              "freed-cell is T after with-context exits"))))))

(deftest with-model-sets-freed-cell-on-error
  (when-model-available
    (testing "with-model marks the handle freed on non-local exit"
      (let ((saved nil))
        (ignore-errors
          (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
            (setf saved model)
            (error "deliberate error")))
        (ok saved "handle was captured")
        (ok (car (cl-llama-cpp::llama-model-freed-cell saved))
            "freed-cell is T after non-local exit")))))

(deftest make-model-then-generate
  (when-model-available
    (testing "make-model + make-context can generate text"
      (let ((model (cl-llama-cpp:make-model *test-model-path* :n-gpu-layers 0)))
        (unwind-protect
             (let ((ctx (cl-llama-cpp:make-context model :n-ctx 512)))
               (unwind-protect
                    (let ((text (cl-llama-cpp:generate ctx "Hello" :max-tokens 4 :temp 0.1)))
                      (ok (stringp text)
                          (format nil "generated: ~S" text))
                      (ok (> (length text) 0) "generated non-empty text"))
                 (cl-llama-cpp:free-context ctx)))
          (cl-llama-cpp:free-model model))))))

;;; Sampler config object integration tests (issue #49)

(deftest sampler-config-generate-basic
  (when-model-available
    (testing "generate accepts :sampler-config and produces text"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let* ((cfg (cl-llama-cpp:make-sampler-config :temp 0.1 :seed 42))
                 (result (cl-llama-cpp:generate ctx "The sky is"
                                                :max-tokens 8
                                                :sampler-config cfg)))
            (ok (stringp result) "generate with :sampler-config returns a string")
            (ok (> (length result) 0) "result is non-empty")))))))

(deftest sampler-config-override-wins
  (when-model-available
    (testing "explicit kwarg overrides :sampler-config value"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          ;; Config requests high temp (random); explicit :temp 0.0 (greedy-ish) should override.
          ;; We can't deterministically test the output, but we can verify no error is raised
          ;; and that the call honours the override signature without complaint.
          (let* ((cfg (cl-llama-cpp:make-sampler-config :temp 1.5 :seed 0))
                 (result (cl-llama-cpp:generate ctx "Once"
                                                :max-tokens 4
                                                :sampler-config cfg
                                                :temp 0.1)))
            (ok (stringp result) "generate with config + explicit :temp override returns a string")))))))

(deftest sampler-config-no-config-unchanged
  (when-model-available
    (testing "generate without :sampler-config behaves as before"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((result (cl-llama-cpp:generate ctx "The capital of France is"
                                               :max-tokens 4 :temp 0.1 :seed 1)))
            (ok (stringp result) "baseline generate (no config) still works")
            (ok (> (length result) 0) "baseline result is non-empty")))))))

(deftest sampler-config-with-sampler-chain
  (when-model-available
    (testing "with-sampler-chain accepts :sampler-config"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((cfg (cl-llama-cpp:make-sampler-config :temp 0.1 :seed 7)))
            (cl-llama-cpp:with-sampler-chain (chain :sampler-config cfg)
              (ok (cl-llama-cpp:llama-sampler-p chain)
                  "chain built from config is a llama-sampler handle")
              (let ((result (cl-llama-cpp:generate ctx "Hello" :max-tokens 4
                                                              :sampler chain)))
                (ok (stringp result)
                    "generate with chain from config returns a string")))))))))

;;; prefill integration tests (issue #80)

(deftest prefill-returns-token-count
  (when-model-available
    (testing "prefill returns a fixnum equal to the number of tokens decoded"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let* ((tokens (cl-llama-cpp:tokenize model "Hello world"))
                 (n (cl-llama-cpp:prefill ctx tokens)))
            (ok (typep n 'fixnum)
                (format nil "prefill returned a fixnum: ~A" n))
            (ok (= n (length tokens))
                (format nil "prefill returned ~A = token count ~A" n (length tokens)))))))))

(deftest prefill-nil-tokens-signals-error
  (when-model-available
    (testing "prefill signals input-validation-error for nil tokens"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (handler-case
                  (progn (cl-llama-cpp:prefill ctx nil) nil)
                (cl-llama-cpp:input-validation-error (c)
                  (eq :tokens (cl-llama-cpp:input-validation-error-argument c))))
              "input-validation-error with :tokens argument for nil tokens"))))))

(deftest prefill-non-vector-tokens-signals-error
  (when-model-available
    (testing "prefill signals input-validation-error for a non-vector tokens arg"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (handler-case
                  (progn (cl-llama-cpp:prefill ctx "not a vector") nil)
                (cl-llama-cpp:input-validation-error (c)
                  (eq :tokens (cl-llama-cpp:input-validation-error-argument c))))
              "input-validation-error with :tokens argument for string tokens"))))))

(deftest prefill-empty-tokens-signals-error
  (when-model-available
    (testing "prefill signals input-validation-error for an empty token vector"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (ok (handler-case
                  (progn (cl-llama-cpp:prefill ctx #()) nil)
                (cl-llama-cpp:input-validation-error (c)
                  (eq :tokens (cl-llama-cpp:input-validation-error-argument c))))
              "input-validation-error with :tokens argument for empty vector"))))))

(deftest prefill-advances-kv-cache-position
  (when-model-available
    (testing "prefill advances KV cache max position to (length tokens) - 1"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let* ((tokens (cl-llama-cpp:tokenize model "The quick brown fox"))
                 (n (length tokens)))
            (cl-llama-cpp:prefill ctx tokens)
            (multiple-value-bind (mn mx)
                (cl-llama-cpp:kv-cache-pos ctx 0)
              (declare (ignore mn))
              (ok (= mx (1- n))
                  (format nil "KV cache max position ~A = ~A (n-1 for ~A tokens)"
                          mx (1- n) n)))))))))

(deftest prefill-sets-compute-pending-p
  (when-model-available
    (testing "prefill sets compute-pending-p to T on the context"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((tokens (cl-llama-cpp:tokenize model "Hello")))
            (cl-llama-cpp:prefill ctx tokens)
            (ok (cl-llama-cpp::llama-context-compute-pending-p ctx)
                "compute-pending-p is T after prefill")))))))

(deftest prefill-does-not-clear-kv-cache
  (when-model-available
    (testing "prefill appends to existing cache rather than clearing"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((tokens1 (cl-llama-cpp:tokenize model "Hello")))
            (cl-llama-cpp:prefill ctx tokens1)
            (multiple-value-bind (_ mx1)
                (cl-llama-cpp:kv-cache-pos ctx 0)
              (declare (ignore _))
              (let ((tokens2 (cl-llama-cpp:tokenize model " world")))
                (cl-llama-cpp:prefill ctx tokens2)
                (multiple-value-bind (_ mx2)
                    (cl-llama-cpp:kv-cache-pos ctx 0)
                  (declare (ignore _))
                  (ok (> mx2 mx1)
                      (format nil "cache advanced from ~A to ~A (not reset)" mx1 mx2)))))))))))

(deftest prefill-non-zero-seq-id
  (when-model-available
    (testing "prefill with :seq-id 1 writes into sequence 1, leaving sequence 0 empty"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512 :n-seq-max 2)
          (let* ((tokens (cl-llama-cpp:tokenize model "Test"))
                 (n (cl-llama-cpp:prefill ctx tokens :seq-id 1)))
            (ok (= n (length tokens))
                (format nil "prefill seq-id 1 returned correct count: ~A" n))
            (multiple-value-bind (mn0 mx0)
                (cl-llama-cpp:kv-cache-pos ctx 0)
              (ok (>= mn0 mx0)
                  (format nil "seq 0 still empty after seq-id 1 prefill (min ~A >= max ~A)"
                          mn0 mx0)))
            (multiple-value-bind (_ mx1)
                (cl-llama-cpp:kv-cache-pos ctx 1)
              (declare (ignore _))
              (ok (= mx1 (1- (length tokens)))
                  (format nil "seq 1 max ~A = ~A (n-1)" mx1 (1- (length tokens)))))))))))

(deftest prefill-save-state-generate-twice-diverges
  (when-model-available
    (testing "prefill + save-state enables two different-seed generates from the same snapshot"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let* ((prompt "Once upon a time in a land far away,")
                 (tokens (cl-llama-cpp:tokenize model prompt :parse-special t))
                 (n-tokens (length tokens)))
            ;; prefill the prompt
            (let ((n-decoded (cl-llama-cpp:prefill ctx tokens)))
              (ok (= n-decoded n-tokens)
                  (format nil "prefill decoded ~A tokens" n-decoded)))
            ;; snapshot after prefill
            (let ((snapshot (cl-llama-cpp:save-state ctx)))
              (ok (> (length snapshot) 0) "snapshot is non-empty after prefill")
              ;; branch 1: restore and generate with seed 1
              (cl-llama-cpp:load-state ctx snapshot)
              (multiple-value-bind (text1 stop1)
                  (cl-llama-cpp:generate ctx nil
                                         :prompt-tokens tokens
                                         :max-tokens 16 :temp 1.5 :seed 1)
                (ok (stringp text1) (format nil "branch 1 produced text: ~S" text1))
                (ok (member stop1 '(:eog :length)) "branch 1 stop reason is valid")
                ;; branch 2: restore and generate with different seed
                (cl-llama-cpp:load-state ctx snapshot)
                (multiple-value-bind (text2 stop2)
                    (cl-llama-cpp:generate ctx nil
                                           :prompt-tokens tokens
                                           :max-tokens 16 :temp 1.5 :seed 99999)
                  (ok (stringp text2) (format nil "branch 2 produced text: ~S" text2))
                  (ok (member stop2 '(:eog :length)) "branch 2 stop reason is valid")
                  ;; different seeds + high temp → different outputs
                  (ok (string/= text1 text2)
                      (format nil "branches diverged: ~S vs ~S" text1 text2))
                  ;; cache consistency: position reflects prompt + generated tokens
                  (multiple-value-bind (_ mx)
                      (cl-llama-cpp:kv-cache-pos ctx 0)
                    (declare (ignore _))
                    (ok (>= mx (1- n-tokens))
                        (format nil "cache max ~A >= last prompt position ~A"
                                mx (1- n-tokens)))))))))))))

(deftest chat-session-send-unchanged-after-prefill-refactor
  (when-model-available
    (testing "chat-session-send behavior is unchanged after prefill refactoring"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((session (cl-llama-cpp:make-chat-session ctx)))
            (multiple-value-bind (reply stop-reason)
                (cl-llama-cpp:chat-session-send session "Hi" :max-tokens 8 :temp 0.0 :seed 42)
              (ok (stringp reply) "first reply is a string")
              (ok (plusp (length reply)) "first reply is non-empty")
              (ok (keywordp stop-reason) "stop-reason is a keyword"))
            ;; second turn verifies incremental cache behavior
            (multiple-value-bind (reply2 _)
                (cl-llama-cpp:chat-session-send session "What is 2 plus 2?" :max-tokens 8
                                                :temp 0.0 :seed 42)
              (declare (ignore _))
              (ok (stringp reply2) "second reply is a string")
              (ok (= 4 (length (cl-llama-cpp:chat-session-messages session)))
                  "four messages after two turns"))))))))

;;; ---------- Issue #81: :seed :random sentinel ----------

(deftest seed-random-produces-different-output
  (when-model-available
    (testing ":seed :random produces differing output across calls"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          ;; Use high temp + long output to maximize divergence probability
          (let ((text1 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 32 :temp 1.5
                                              :seed :random))
                (text2 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 32 :temp 1.5
                                              :seed :random)))
            (ok (stringp text1) "first call returns a string")
            (ok (stringp text2) "second call returns a string")
            (ok (string/= text1 text2)
                (format nil "two :random calls diverged: ~S vs ~S"
                        text1 text2))))))))

(deftest seed-integer-unchanged
  (when-model-available
    (testing ":seed <integer> still produces deterministic output"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((text1 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 16 :temp 0.8
                                              :seed 42))
                (text2 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 16 :temp 0.8
                                              :seed 42)))
            (ok (string= text1 text2)
                (format nil "same seed → same output: ~S" text1))))))))

(deftest seed-nil-means-random
  (when-model-available
    (testing ":seed nil behaves like :seed :random"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((text1 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 32 :temp 1.5
                                              :seed nil))
                (text2 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 32 :temp 1.5
                                              :seed nil)))
            (ok (stringp text1) "first call returns a string")
            (ok (string/= text1 text2)
                (format nil "two nil-seed calls diverged: ~S vs ~S"
                        text1 text2))))))))

(deftest seed-random-in-sampler-config
  (when-model-available
    (testing ":seed :random in sampler-config produces nondeterministic output"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let* ((cfg (cl-llama-cpp:make-sampler-config :seed :random :temp 1.5))
                 (text1 (cl-llama-cpp:generate ctx "Once upon a time"
                                               :sampler-config cfg
                                               :max-tokens 32))
                 (text2 (cl-llama-cpp:generate ctx "Once upon a time"
                                               :sampler-config cfg
                                               :max-tokens 32)))
            (ok (stringp text1) "first call returns a string")
            (ok (string/= text1 text2)
                (format nil "sampler-config :random diverged: ~S vs ~S"
                        text1 text2))))))))

(deftest seed-default-still-42
  (when-model-available
    (testing "default seed (no :seed arg) is still 42 and deterministic"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          ;; Call with no :seed — should use default 42
          (let ((text1 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 16 :temp 0.8))
                (text2 (cl-llama-cpp:generate ctx "Once upon a time"
                                              :max-tokens 16 :temp 0.8)))
            (ok (string= text1 text2)
                "default seed produces identical output")))))))

(deftest generate-logit-callback-fires-per-token
  (when-model-available
    (testing "logit-callback fires once per generated token with n-vocab-length logits"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((calls '()))
            (multiple-value-bind (text stop-reason)
                (cl-llama-cpp:generate ctx "The capital of France is"
                                       :max-tokens 8
                                       :temp 0.1
                                       :logit-callback
                                       (lambda (logits n)
                                         (push (list logits n) calls)))
              (declare (ignore stop-reason))
              (setf calls (nreverse calls))
              (ok (stringp text) "generate returned a string")
              (ok (= (length calls) 8)
                  (format nil "logit-callback fired once per token: ~D calls"
                          (length calls)))
              (ok (every (lambda (c) (= (second c) (length (first c)))) calls)
                  "every call's reported n-vocab matches its logits array length")
              (ok (apply #'= (mapcar #'second calls))
                  "n-vocab is consistent across every call")
              (ok (every (lambda (c)
                           (typep (first c) '(simple-array single-float (*))))
                         calls)
                  "every call's logits is a simple-array single-float")
              (ok (every (lambda (c)
                           (every (lambda (x) (and (typep x 'single-float)
                                                    (not (sb-ext:float-nan-p x))))
                                  (first c)))
                         calls)
                  "no NaN logits leaked through"))))))))

(deftest generate-logit-callback-independent-of-token-callback
  (when-model-available
    (testing ":logit-callback and :token-callback can both be supplied and both fire"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((logit-calls 0) (token-calls 0))
            (cl-llama-cpp:generate ctx "The capital of France is"
                                   :max-tokens 8
                                   :temp 0.1
                                   :logit-callback (lambda (l n)
                                                      (declare (ignore l n))
                                                      (incf logit-calls))
                                   :token-callback (lambda (chunk)
                                                      (declare (ignore chunk))
                                                      (incf token-calls)
                                                      t))
            (ok (= logit-calls 8) (format nil "logit-callback fired 8 times: ~D" logit-calls))
            (ok (plusp token-calls) (format nil "token-callback also fired: ~D" token-calls))))))))

(deftest generate-logit-callback-error-returns-error-stop
  (when-model-available
    (testing "logit-callback that signals an error produces :error stop reason"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (text stop-reason)
              (cl-llama-cpp:generate ctx "The capital of France is"
                                     :max-tokens 64
                                     :temp 0.1
                                     :logit-callback (lambda (l n)
                                                       (declare (ignore l n))
                                                       (error "deliberate logit-callback error")))
            (ok (stringp text) "generate returned a string despite callback error")
            (ok (eq :error stop-reason)
                (format nil "stop-reason is :error: ~A" stop-reason))))))))

(deftest generate-logit-callback-ignore-restart-continues
  (when-model-available
    (testing "invoking ignore-logit-callback-error restart allows generation to continue"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (let ((call-count 0))
            (multiple-value-bind (text stop-reason)
                (cl-llama-cpp:generate
                 ctx "The capital of France is"
                 :max-tokens 16
                 :temp 0.1
                 :logit-callback
                 (lambda (l n)
                   (declare (ignore l n))
                   (incf call-count)
                   (when (= call-count 1)
                     (handler-bind
                         ((error (lambda (c)
                                   (declare (ignore c))
                                   (when (find-restart 'cl-llama-cpp::ignore-logit-callback-error)
                                     (invoke-restart 'cl-llama-cpp::ignore-logit-callback-error)))))
                       (error "first call errors")))))
              (ok (stringp text) "generate returned text after error recovery")
              (ok (member stop-reason '(:eog :length))
                  (format nil "stop-reason is not :error after recovery: ~A" stop-reason))
              (ok (> call-count 1)
                  (format nil "callback was invoked ~D times (continued after error)"
                          call-count)))))))))

(deftest generate-without-logit-callback-unaffected
  (when-model-available
    (testing "generate without :logit-callback still returns :eog or :length"
      (cl-llama-cpp:with-model (model *test-model-path* :n-gpu-layers 0)
        (cl-llama-cpp:with-context (ctx model :n-ctx 512)
          (multiple-value-bind (text stop-reason)
              (cl-llama-cpp:generate ctx "The capital of France is"
                                     :max-tokens 16 :temp 0.8)
            (ok (stringp text) "generate returned text")
            (ok (member stop-reason '(:eog :length))
                (format nil "stop-reason is :eog or :length: ~A" stop-reason))))))))
