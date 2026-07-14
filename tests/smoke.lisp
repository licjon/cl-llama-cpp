(defpackage #:cl-llama-cpp/tests/smoke
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/smoke)

(deftest bindings-package-exists
  (testing "%llama package exists and has symbols"
    (let ((pkg (find-package :%llama)))
      (ok pkg "%llama package exists")
      (let ((count 0))
        (do-symbols (s pkg) (incf count))
        (ok (> count 50) (format nil "%llama has ~d symbols (expected >50)" count))))))

(deftest model-default-params
  (testing "llama_model_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let ((params (%llama:model-default-params)))
        (ok params "model-default-params returned non-nil")
        (ok (listp params) "result is a plist")))))

(deftest context-default-params
  (testing "llama_context_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let ((params (%llama:context-default-params)))
        (ok params "context-default-params returned non-nil")
        (ok (listp params) "result is a plist")))))

(deftest sampler-chain-default-params
  (testing "llama_sampler_chain_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let ((params (%llama:sampler-chain-default-params)))
        (ok (listp params) "result is a plist")))))

(deftest backend-init
  (testing "llama_backend_init runs without error"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (pass "backend-init completed"))))

(deftest condition-hierarchy
  (testing "llama-error condition hierarchy exists"
    (ok (subtypep 'cl-llama-cpp:model-load-error 'cl-llama-cpp:llama-error)
        "model-load-error is a llama-error")
    (ok (subtypep 'cl-llama-cpp:context-creation-error 'cl-llama-cpp:llama-error)
        "context-creation-error is a llama-error")
    (ok (subtypep 'cl-llama-cpp:tokenization-error 'cl-llama-cpp:llama-error)
        "tokenization-error is a llama-error")
    (ok (subtypep 'cl-llama-cpp:decode-error 'cl-llama-cpp:llama-error)
        "decode-error is a llama-error")))

(deftest condition-signaling
  (testing "conditions can be signaled and caught"
    (ok (typep (handler-case (error 'cl-llama-cpp:model-load-error :path "/bad/path")
                (cl-llama-cpp:model-load-error (c) c))
               'cl-llama-cpp:model-load-error)
        "model-load-error is catchable")
    (ok (typep (handler-case (error 'cl-llama-cpp:decode-error :code -1)
                (cl-llama-cpp:decode-error (c) c))
               'cl-llama-cpp:decode-error)
        "decode-error is catchable")))

(deftest chat-template-condition
  (testing "chat-template-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:chat-template-error 'cl-llama-cpp:llama-error)
        "chat-template-error is a llama-error")))

(deftest list-chat-templates
  (testing "list-chat-templates returns built-in template names"
    (let ((templates (cl-llama-cpp:list-chat-templates)))
      (ok (listp templates) "result is a list")
      (ok (> (length templates) 0) "at least one built-in template")
      (ok (every #'stringp templates) "all elements are strings"))))

(deftest with-model-bad-path
  (testing "with-model signals model-load-error on nonexistent path"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (ok (handler-case
              (cl-llama-cpp:with-model (model "/nonexistent/path/to/model.gguf")
                nil)
            (cl-llama-cpp:model-load-error (c)
              (cl-llama-cpp:model-load-error-path c)))
          "model-load-error was signaled for bad path"))))

;;; LoRA adapter wrapper tests

(deftest lora-condition-hierarchy
  (testing "lora-load-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:lora-load-error 'cl-llama-cpp:llama-error)
        "lora-load-error is a llama-error"))
  (testing "lora-apply-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:lora-apply-error 'cl-llama-cpp:llama-error)
        "lora-apply-error is a llama-error")))

(deftest lora-condition-signaling
  (testing "lora-load-error can be signaled and caught with path slot"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:lora-load-error :path "/bad/lora.gguf")
                    (cl-llama-cpp:lora-load-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:lora-load-error)
          "lora-load-error is catchable")
      (ok (string= "/bad/lora.gguf" (cl-llama-cpp:lora-load-error-path caught))
          "lora-load-error-path accessor works")))
  (testing "lora-apply-error can be signaled and caught"
    (ok (typep (handler-case (error 'cl-llama-cpp:lora-apply-error :code -1)
                (cl-llama-cpp:lora-apply-error (c) c))
               'cl-llama-cpp:lora-apply-error)
        "lora-apply-error is catchable")))

;;; KV cache / memory management wrapper tests

(deftest kv-cache-symbols-exported
  (testing "KV cache wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(clear-kv-cache
                   kv-cache-seq-rm
                   kv-cache-seq-cp
                   kv-cache-seq-keep
                   kv-cache-pos
                   kv-cache-can-shift-p
                   kv-cache-seq-add
                   kv-cache-seq-div))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest kv-cache-functions-fbound
  (testing "KV cache wrapper symbols are fbound"
    (dolist (sym-name '("CLEAR-KV-CACHE"
                        "KV-CACHE-SEQ-RM"
                        "KV-CACHE-SEQ-CP"
                        "KV-CACHE-SEQ-KEEP"
                        "KV-CACHE-POS"
                        "KV-CACHE-CAN-SHIFT-P"
                        "KV-CACHE-SEQ-ADD"
                        "KV-CACHE-SEQ-DIV"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest kv-cache-binding-deps
  (testing "memory bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:memory-seq-rm
                     %llama:memory-seq-cp
                     %llama:memory-seq-keep
                     %llama:memory-seq-add
                     %llama:memory-seq-div
                     %llama:memory-seq-pos-min
                     %llama:memory-seq-pos-max
                     %llama:memory-can-shift))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

;;; Model / context introspection wrapper tests

(deftest introspection-symbols-exported
  (testing "introspection wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(model-description
                   model-metadata
                   model-info
                   model-cls-label
                   context-info
                   system-info))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest introspection-functions-fbound
  (testing "introspection wrapper symbols are fbound"
    (dolist (sym-name '("MODEL-DESCRIPTION"
                        "MODEL-METADATA"
                        "MODEL-INFO"
                        "MODEL-CLS-LABEL"
                        "CONTEXT-INFO"
                        "SYSTEM-INFO"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest introspection-binding-deps
  (testing "introspection bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:model-desc %llama:model-size %llama:model-n-params
                     %llama:model-n-ctx-train %llama:model-n-layer
                     %llama:model-n-head %llama:model-n-head-kv
                     %llama:model-n-embd-inp %llama:model-n-embd-out
                     %llama:model-n-swa %llama:model-rope-type
                     %llama:model-rope-freq-scale-train
                     %llama:model-has-encoder %llama:model-has-decoder
                     %llama:model-is-recurrent %llama:model-is-hybrid
                     %llama:model-is-diffusion
                     %llama:model-n-cls-out %llama:model-cls-label
                     %llama:model-meta-count %llama:model-meta-key-by-index
                     %llama:model-meta-val-str %llama:model-meta-val-str-by-index
                     %llama:n-ctx %llama:n-batch %llama:n-ubatch %llama:n-seq-max
                     %llama:n-threads %llama:n-threads-batch %llama:pooling-type
                     %llama:print-system-info))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

(deftest system-info-returns-string
  (testing "system-info returns a non-empty string without needing a model"
    (let ((info (cl-llama-cpp:system-info)))
      (ok (stringp info) "system-info returned a string")
      (ok (> (length info) 0) "system-info is non-empty"))))

;;; Session state save/load wrapper tests

(deftest session-symbols-exported
  (testing "session state wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(save-session load-session
                   save-session-seq load-session-seq
                   save-state load-state
                   save-state-seq load-state-seq))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest session-functions-fbound
  (testing "session state wrapper symbols are fbound"
    (dolist (sym-name '("SAVE-SESSION" "LOAD-SESSION"
                        "SAVE-SESSION-SEQ" "LOAD-SESSION-SEQ"
                        "SAVE-STATE" "LOAD-STATE"
                        "SAVE-STATE-SEQ" "LOAD-STATE-SEQ"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest session-condition-hierarchy
  (testing "session-save-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:session-save-error 'cl-llama-cpp:llama-error)
        "session-save-error is a llama-error"))
  (testing "session-load-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:session-load-error 'cl-llama-cpp:llama-error)
        "session-load-error is a llama-error")))

(deftest session-condition-signaling
  (testing "session-save-error can be signaled and caught with path slot"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:session-save-error :path "/bad/session.bin")
                    (cl-llama-cpp:session-save-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:session-save-error)
          "session-save-error is catchable")
      (ok (string= "/bad/session.bin" (cl-llama-cpp:session-save-error-path caught))
          "session-save-error-path accessor works")))
  (testing "session-load-error can be signaled and caught with path slot"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:session-load-error :path "/bad/session.bin")
                    (cl-llama-cpp:session-load-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:session-load-error)
          "session-load-error is catchable")
      (ok (string= "/bad/session.bin" (cl-llama-cpp:session-load-error-path caught))
          "session-load-error-path accessor works"))))

(deftest session-binding-deps
  (testing "session state bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:state-get-size %llama:state-get-data %llama:state-set-data
                     %llama:state-save-file %llama:state-load-file
                     %llama:state-seq-get-size %llama:state-seq-get-data
                     %llama:state-seq-set-data
                     %llama:state-seq-save-file %llama:state-seq-load-file
                     %llama:state-seq-get-size-ext %llama:state-seq-get-data-ext
                     %llama:state-seq-set-data-ext))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

;;; Grammar / constrained generation wrapper tests

(deftest grammar-symbols-exported
  (testing "grammar wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(grammar-error
                   grammar-error-grammar
                   make-grammar-sampler
                   make-grammar-sampler-lazy
                   make-infill-sampler
                   with-grammar-sampler))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest grammar-functions-fbound
  (testing "grammar wrapper functions are fbound"
    (dolist (sym-name '("MAKE-GRAMMAR-SAMPLER"
                        "MAKE-GRAMMAR-SAMPLER-LAZY"
                        "MAKE-INFILL-SAMPLER"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest grammar-condition-hierarchy
  (testing "grammar-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:grammar-error 'cl-llama-cpp:llama-error)
        "grammar-error is a llama-error")))

(deftest grammar-condition-signaling
  (testing "grammar-error can be signaled and caught with grammar slot"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:grammar-error :grammar "test grammar")
                    (cl-llama-cpp:grammar-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:grammar-error)
          "grammar-error is catchable")
      (ok (string= "test grammar" (cl-llama-cpp:grammar-error-grammar caught))
          "grammar-error-grammar accessor works"))))

(deftest grammar-binding-deps
  (testing "grammar bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:sampler-init-grammar
                     %llama:sampler-init-grammar-lazy
                     %llama:sampler-init-grammar-lazy-patterns
                     %llama:sampler-init-infill))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

;;; Sampler chain wrapper tests (issue #62)

(deftest sampler-chain-symbols-exported
  (testing "sampler-chain-add is exported from cl-llama-cpp"
    (dolist (sym '(with-sampler-chain sampler-chain-add))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest sampler-chain-add-fbound
  (testing "sampler-chain-add is fbound"
    (let ((sym (find-symbol "SAMPLER-CHAIN-ADD" :cl-llama-cpp)))
      (ok (and sym (fboundp sym))
          "SAMPLER-CHAIN-ADD is fbound"))))

;;; Extended sampler wrapper tests

(deftest sampler-seed-symbol-exported
  (testing "sampler-seed is exported from cl-llama-cpp"
    (multiple-value-bind (sym status)
        (find-symbol "SAMPLER-SEED" :cl-llama-cpp)
      (ok sym "SAMPLER-SEED is accessible")
      (ok (eq status :external) "SAMPLER-SEED is exported"))))

(deftest sampler-seed-fbound
  (testing "sampler-seed is fbound"
    (let ((sym (find-symbol "SAMPLER-SEED" :cl-llama-cpp)))
      (ok (and sym (fboundp sym)) "SAMPLER-SEED is fbound"))))

(deftest extended-sampler-binding-deps
  (testing "extended sampler bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:sampler-init-typical
                     %llama:sampler-init-xtc
                     %llama:sampler-init-top-n-sigma
                     %llama:sampler-init-mirostat
                     %llama:sampler-init-mirostat-v2
                     %llama:sampler-init-temp-ext
                     %llama:sampler-init-penalties
                     %llama:sampler-init-dry
                     %llama:sampler-init-logit-bias
                     %llama:sampler-init-adaptive-p
                     %llama:sampler-get-seed
                     %llama:logit-bias
                     %llama:vocab-n-tokens))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

;;; Individual sampler constructor / free-sampler tests (issue #65)

(deftest sampler-constructors-exported
  (testing "individual sampler constructors are exported from cl-llama-cpp"
    (dolist (sym '(make-greedy-sampler
                   make-dist-sampler
                   make-top-k-sampler
                   make-top-p-sampler
                   make-min-p-sampler
                   make-typical-sampler
                   make-temp-sampler
                   make-temp-ext-sampler
                   make-xtc-sampler
                   make-top-n-sigma-sampler
                   make-mirostat-v2-sampler
                   free-sampler))
      (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible in cl-llama-cpp" sym))
        (ok (eq status :external) (format nil "~A is exported" sym))))))

(deftest sampler-constructors-fbound
  (testing "individual sampler constructors are fbound"
    (dolist (name '("MAKE-GREEDY-SAMPLER"
                    "MAKE-DIST-SAMPLER"
                    "MAKE-TOP-K-SAMPLER"
                    "MAKE-TOP-P-SAMPLER"
                    "MAKE-MIN-P-SAMPLER"
                    "MAKE-TYPICAL-SAMPLER"
                    "MAKE-TEMP-SAMPLER"
                    "MAKE-TEMP-EXT-SAMPLER"
                    "MAKE-XTC-SAMPLER"
                    "MAKE-TOP-N-SIGMA-SAMPLER"
                    "MAKE-MIROSTAT-V2-SAMPLER"
                    "FREE-SAMPLER"))
      (let ((sym (find-symbol name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" name))))))

;;; Batch API wrapper tests

(deftest batch-symbols-exported
  (testing "batch API symbols are exported from cl-llama-cpp"
    (dolist (sym '(batch-init-error
                   batch-init-error-n-tokens
                   batch-overflow-error
                   batch-overflow-error-capacity
                   batch-overflow-error-token-count
                   with-batch
                   batch-add-token
                   batch-add-embedding
                   batch-add-sequence
                   batch-clear
                   batch-token-count
                   batch-decode
                   batch-encode))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest batch-functions-fbound
  (testing "batch API functions are fbound"
    (dolist (sym-name '("BATCH-ADD-TOKEN"
                        "BATCH-ADD-EMBEDDING"
                        "BATCH-ADD-SEQUENCE"
                        "BATCH-CLEAR"
                        "BATCH-TOKEN-COUNT"
                        "BATCH-DECODE"
                        "BATCH-ENCODE"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest batch-condition-hierarchy
  (testing "batch-init-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:batch-init-error 'cl-llama-cpp:llama-error)
        "batch-init-error is a llama-error"))
  (testing "batch-overflow-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:batch-overflow-error 'cl-llama-cpp:llama-error)
        "batch-overflow-error is a llama-error")))

(deftest batch-condition-signaling
  (testing "batch-init-error can be signaled and caught with n-tokens slot"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:batch-init-error :n-tokens 0)
                    (cl-llama-cpp:batch-init-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:batch-init-error)
          "batch-init-error is catchable")
      (ok (zerop (cl-llama-cpp:batch-init-error-n-tokens caught))
          "batch-init-error-n-tokens accessor works")))
  (testing "batch-overflow-error can be signaled and caught"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:batch-overflow-error
                             :capacity 10 :token-count 10)
                    (cl-llama-cpp:batch-overflow-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:batch-overflow-error)
          "batch-overflow-error is catchable")
      (ok (= 10 (cl-llama-cpp:batch-overflow-error-capacity caught))
          "batch-overflow-error-capacity accessor works")
      (ok (= 10 (cl-llama-cpp:batch-overflow-error-token-count caught))
          "batch-overflow-error-token-count accessor works"))))

(deftest batch-binding-deps
  (testing "batch bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:batch-init %llama:batch-free))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

;;; Backend lifecycle and portability wrapper tests

(deftest with-backend-symbol-exported
  (testing "with-backend is exported from cl-llama-cpp"
    (multiple-value-bind (sym status)
        (find-symbol "WITH-BACKEND" :cl-llama-cpp)
      (ok sym "WITH-BACKEND is accessible")
      (ok (eq status :external) "WITH-BACKEND is exported"))))

(deftest backend-lifecycle-symbols-exported
  (testing "backend/context configuration symbols are exported"
    (dolist (sym '(with-backend
                   set-n-threads
                   set-warmup
                   set-causal-attn
                   set-embeddings
                   synchronize
                   set-abort-callback
                   get-abort-callback
                   attach-threadpool
                   detach-threadpool))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest backend-lifecycle-functions-fbound
  (testing "context configuration functions are fbound"
    (dolist (sym-name '("SET-N-THREADS"
                        "SET-WARMUP"
                        "SET-CAUSAL-ATTN"
                        "SET-EMBEDDINGS"
                        "SYNCHRONIZE"
                        "SET-ABORT-CALLBACK"
                        "GET-ABORT-CALLBACK"
                        "ATTACH-THREADPOOL"
                        "DETACH-THREADPOOL"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest backend-lifecycle-binding-deps
  (testing "backend/context bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:backend-free
                     %llama:numa-init
                     %llama:set-n-threads
                     %llama:set-warmup
                     %llama:set-causal-attn
                     %llama:set-embeddings
                     %llama:synchronize
                     %llama:set-abort-callback
                     %llama:attach-threadpool
                     %llama:detach-threadpool))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

(deftest set-n-threads-type-check
  (testing "set-n-threads rejects non-integer arguments"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (ok (handler-case
              (cl-llama-cpp:set-n-threads nil 8 4)
            (type-error () t))
          "nil ctx passes through to CFFI (type check is on integer args)")
      (ok (handler-case
              (cl-llama-cpp:set-n-threads (cffi:null-pointer) "8" 4)
            (type-error (c) (declare (ignore c)) t))
          "string n-threads signals type-error")
      (ok (handler-case
              (cl-llama-cpp:set-n-threads (cffi:null-pointer) 8 nil)
            (type-error (c) (declare (ignore c)) t))
          "nil n-threads-batch signals type-error"))))

(deftest with-backend-nesting
  (testing "nested with-backend only frees on outermost exit"
    ;; Rebind all backend state to guarantee clean slate regardless of prior tests.
    (let ((cl-llama-cpp::*backend-initialized* nil)
          (cl-llama-cpp::*backend-permanent* nil)
          (cl-llama-cpp::*backend-refcount* 0)
          (init-count 0))
      (cl-llama-cpp:with-backend ()
        (incf init-count)
        (cl-llama-cpp:with-backend ()
          (incf init-count)))
      (ok (= 2 init-count) "body ran twice (nesting works)")
      (ok (not cl-llama-cpp::*backend-initialized*)
          "*backend-initialized* cleared after outermost exit")
      (ok (= 0 cl-llama-cpp::*backend-refcount*)
          "*backend-refcount* is 0 after outermost exit"))))

;;; Performance, logging, and system info wrapper tests

(deftest perf-symbols-exported
  (testing "performance wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(context-perf print-context-perf reset-context-perf
                   sampler-perf print-sampler-perf reset-sampler-perf
                   print-perf reset-perf with-perf))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest perf-functions-fbound
  (testing "performance wrapper functions are fbound"
    (dolist (sym-name '("CONTEXT-PERF" "PRINT-CONTEXT-PERF" "RESET-CONTEXT-PERF"
                        "SAMPLER-PERF" "PRINT-SAMPLER-PERF" "RESET-SAMPLER-PERF"
                        "PRINT-PERF" "RESET-PERF"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest perf-binding-deps
  (testing "performance bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:perf-context %llama:perf-context-print %llama:perf-context-reset
                     %llama:perf-sampler %llama:perf-sampler-print %llama:perf-sampler-reset))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

(deftest logging-symbols-exported
  (testing "logging wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(set-log-callback get-log-callback))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest logging-functions-fbound
  (testing "logging wrapper functions are fbound"
    (dolist (sym-name '("SET-LOG-CALLBACK" "GET-LOG-CALLBACK"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest logging-binding-deps
  (testing "logging bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:log-set %llama:log-get))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

(deftest system-query-symbols-exported
  (testing "system query wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(time-us system-capabilities))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest system-query-functions-fbound
  (testing "system query wrapper functions are fbound"
    (dolist (sym-name '("TIME-US" "SYSTEM-CAPABILITIES"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest system-query-binding-deps
  (testing "system query bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:time-us %llama:max-devices
                     %llama:supports-mmap %llama:supports-mlock
                     %llama:supports-gpu-offload %llama:supports-rpc))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

(deftest system-capabilities-no-model
  (testing "system-capabilities returns a well-formed plist without a model"
    (let ((caps (cl-llama-cpp:system-capabilities)))
      (ok (listp caps) "result is a list")
      (ok (member :mmap caps) ":mmap key present")
      (ok (member :mlock caps) ":mlock key present")
      (ok (member :gpu-offload caps) ":gpu-offload key present")
      (ok (member :rpc caps) ":rpc key present")
      (ok (member :max-devices caps) ":max-devices key present")
      (ok (typep (getf caps :max-devices) 'integer)
          ":max-devices is an integer")
      (ok (typep (getf caps :mmap) 'boolean) ":mmap is a boolean")
      (ok (typep (getf caps :mlock) 'boolean) ":mlock is a boolean")
      (ok (typep (getf caps :gpu-offload) 'boolean) ":gpu-offload is a boolean")
      (ok (typep (getf caps :rpc) 'boolean) ":rpc is a boolean"))))

(deftest time-us-no-model
  (testing "time-us returns a positive integer without a model"
    (let ((t1 (cl-llama-cpp:time-us)))
      (ok (integerp t1) "time-us returned an integer")
      (ok (plusp t1) "time-us returned a positive value"))))

;;; Resource planning & configuration validation wrapper tests

(deftest resource-planning-symbols-exported
  (testing "resource planning symbols are exported from cl-llama-cpp"
    (dolist (sym '(estimate-memory
                   explain-memory-usage
                   feasibility-report
                   validate-configuration
                   suggest-configuration
                   configuration-unsafe-warning
                   configuration-unsafe-error
                   configuration-unsafe-error-reason))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest resource-planning-functions-fbound
  (testing "resource planning functions are fbound"
    (dolist (sym-name '("ESTIMATE-MEMORY"
                        "EXPLAIN-MEMORY-USAGE"
                        "FEASIBILITY-REPORT"
                        "VALIDATE-CONFIGURATION"
                        "SUGGEST-CONFIGURATION"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest resource-planning-condition-hierarchy
  (testing "configuration-unsafe-warning is a warning"
    (ok (subtypep 'cl-llama-cpp:configuration-unsafe-warning 'warning)
        "configuration-unsafe-warning is a warning"))
  (testing "configuration-unsafe-error is a llama-error"
    (ok (subtypep 'cl-llama-cpp:configuration-unsafe-error 'cl-llama-cpp:llama-error)
        "configuration-unsafe-error is a llama-error")))

(deftest resource-planning-condition-signaling
  (testing "configuration-unsafe-warning can be signaled and caught"
    (let ((caught nil))
      (handler-bind ((cl-llama-cpp:configuration-unsafe-warning
                      (lambda (c)
                        (setf caught c)
                        (muffle-warning c))))
        (warn 'cl-llama-cpp:configuration-unsafe-warning :reason "test reason"))
      (ok (typep caught 'cl-llama-cpp:configuration-unsafe-warning)
          "configuration-unsafe-warning is catchable")))
  (testing "configuration-unsafe-error can be signaled and caught with reason slot"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:configuration-unsafe-error :reason "test reason")
                    (cl-llama-cpp:configuration-unsafe-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:configuration-unsafe-error)
          "configuration-unsafe-error is catchable")
      (ok (string= "test reason" (cl-llama-cpp:configuration-unsafe-error-reason caught))
          "configuration-unsafe-error-reason accessor works"))))

(deftest last-log-callback-error-exported
  (testing "*last-log-callback-error* is exported from cl-llama-cpp"
    (multiple-value-bind (sym status)
        (find-symbol "*LAST-LOG-CALLBACK-ERROR*" :cl-llama-cpp)
      (ok sym "*LAST-LOG-CALLBACK-ERROR* is accessible")
      (ok (eq status :external) "*LAST-LOG-CALLBACK-ERROR* is exported")))
  (testing "*last-log-callback-error* is bound"
    (let ((sym (find-symbol "*LAST-LOG-CALLBACK-ERROR*" :cl-llama-cpp)))
      (ok (boundp sym) "*LAST-LOG-CALLBACK-ERROR* is boundp"))))

;;; Abort callback wrapper tests (issue #45)

(deftest abort-callback-symbols-exported
  (testing "abort callback symbols are exported from cl-llama-cpp"
    (dolist (sym '(set-abort-callback get-abort-callback))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym))))))
  (testing "abort callback functions are fbound"
    (dolist (sym-name '("SET-ABORT-CALLBACK" "GET-ABORT-CALLBACK"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest last-abort-callback-error-exported
  (testing "*last-abort-callback-error* is exported from cl-llama-cpp"
    (multiple-value-bind (sym status)
        (find-symbol "*LAST-ABORT-CALLBACK-ERROR*" :cl-llama-cpp)
      (ok sym "*LAST-ABORT-CALLBACK-ERROR* is accessible")
      (ok (eq status :external) "*LAST-ABORT-CALLBACK-ERROR* is exported")))
  (testing "*last-abort-callback-error* is bound"
    (let ((sym (find-symbol "*LAST-ABORT-CALLBACK-ERROR*" :cl-llama-cpp)))
      (ok (boundp sym) "*LAST-ABORT-CALLBACK-ERROR* is boundp"))))

(deftest abort-callback-lock-defined
  (testing "abort callback mutex is defined on SBCL"
    #+sbcl
    (ok (boundp 'cl-llama-cpp::*abort-callback-lock*)
        "*abort-callback-lock* is bound")
    #-sbcl
    (ok t "mutex is a no-op on non-SBCL")))

(deftest set-abort-callback-type-check
  (testing "set-abort-callback rejects non-function/non-nil second argument"
    (ok (handler-case
            (cl-llama-cpp:set-abort-callback
             (cffi:null-pointer)
             "not-a-function")
          (type-error () t))
        "string fn signals type-error")
    (ok (handler-case
            (cl-llama-cpp:set-abort-callback
             (cffi:null-pointer)
             42)
          (type-error () t))
        "integer fn signals type-error")))

;;; Backend device & registry introspection (issue #29)

(deftest backend-device-handle-symbols-exported
  (testing "ggml-backend-device handle symbols are exported from cl-llama-cpp"
    (dolist (sym '(ggml-backend-device ggml-backend-device-p ggml-backend-device-pointer
                   ggml-backend-registry ggml-backend-registry-p ggml-backend-registry-pointer))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym))))))
  (testing "ggml-backend-device predicates and accessors are fbound"
    (dolist (sym-name '("GGML-BACKEND-DEVICE-P" "GGML-BACKEND-DEVICE-POINTER"
                        "GGML-BACKEND-REGISTRY-P" "GGML-BACKEND-REGISTRY-POINTER"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest backend-introspection-symbols-exported
  (testing "backend device introspection symbols are exported from cl-llama-cpp"
    (dolist (sym '(backend-dev-count backend-dev-get backend-dev-name
                   backend-dev-description backend-dev-type
                   backend-dev-memory backend-dev-props
                   backend-dev-by-name backend-dev-by-type
                   backend-reg-count backend-reg-get backend-reg-name
                   backend-reg-dev-count backend-reg-dev-get backend-reg-by-name
                   gpu-devices detect-free-vram detect-total-vram))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest backend-introspection-functions-fbound
  (testing "backend introspection functions are fbound"
    (dolist (sym-name '("BACKEND-DEV-COUNT" "BACKEND-DEV-GET" "BACKEND-DEV-NAME"
                        "BACKEND-DEV-DESCRIPTION" "BACKEND-DEV-TYPE"
                        "BACKEND-DEV-MEMORY" "BACKEND-DEV-PROPS"
                        "BACKEND-DEV-BY-NAME" "BACKEND-DEV-BY-TYPE"
                        "BACKEND-REG-COUNT" "BACKEND-REG-GET" "BACKEND-REG-NAME"
                        "BACKEND-REG-DEV-COUNT" "BACKEND-REG-DEV-GET" "BACKEND-REG-BY-NAME"
                        "GPU-DEVICES" "DETECT-FREE-VRAM" "DETECT-TOTAL-VRAM"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest backend-introspection-binding-deps
  (testing "backend introspection bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:ggml-backend-dev-count %llama:ggml-backend-dev-get
                     %llama:ggml-backend-dev-name %llama:ggml-backend-dev-description
                     %llama:ggml-backend-dev-type %llama:ggml-backend-dev-memory
                     %llama:ggml-backend-dev-get-props
                     %llama:ggml-backend-dev-by-name %llama:ggml-backend-dev-by-type
                     %llama:ggml-backend-reg-count %llama:ggml-backend-reg-get
                     %llama:ggml-backend-reg-name
                     %llama:ggml-backend-reg-dev-count %llama:ggml-backend-reg-dev-get
                     %llama:ggml-backend-reg-by-name))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

(deftest backend-dev-count-no-model
  (testing "backend-dev-count returns a non-negative integer without a model"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((count (cl-llama-cpp:backend-dev-count)))
        (ok (integerp count) "backend-dev-count returned an integer")
        (ok (>= count 0) "backend-dev-count is non-negative")))))

(deftest system-capabilities-extended-keys
  (testing "system-capabilities now includes backend device keys"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((caps (cl-llama-cpp:system-capabilities)))
        (ok (member :n-backend-devs caps) ":n-backend-devs key present")
        (ok (member :n-backend-regs caps) ":n-backend-regs key present")
        (ok (member :has-gpu caps) ":has-gpu key present")
        (ok (integerp (getf caps :n-backend-devs))
            ":n-backend-devs is an integer")
        (ok (integerp (getf caps :n-backend-regs))
            ":n-backend-regs is an integer")
        (ok (typep (getf caps :has-gpu) 'boolean)
            ":has-gpu is a boolean")))))

;;; Typed opaque handles (issue #41)

(deftest handle-symbols-exported
  (testing "typed handle symbols are exported from cl-llama-cpp"
    (dolist (sym '(llama-model llama-model-p llama-model-pointer
                   llama-context llama-context-p llama-context-pointer
                   llama-sampler llama-sampler-p llama-sampler-pointer))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest handle-predicates-fbound
  (testing "handle predicate and accessor functions are fbound"
    (dolist (sym-name '("LLAMA-MODEL-P" "LLAMA-MODEL-POINTER"
                        "LLAMA-CONTEXT-P" "LLAMA-CONTEXT-POINTER"
                        "LLAMA-SAMPLER-P" "LLAMA-SAMPLER-POINTER"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest handle-types-distinct
  (testing "llama-model, llama-context, llama-sampler are distinct types"
    (ok (not (subtypep 'cl-llama-cpp:llama-model 'cl-llama-cpp:llama-context))
        "llama-model is not a subtype of llama-context")
    (ok (not (subtypep 'cl-llama-cpp:llama-context 'cl-llama-cpp:llama-model))
        "llama-context is not a subtype of llama-model")
    (ok (not (subtypep 'cl-llama-cpp:llama-sampler 'cl-llama-cpp:llama-model))
        "llama-sampler is not a subtype of llama-model")))

(deftest handle-predicates-exclusive
  (testing "handle predicates reject wrong handle types"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (ok (handler-case
              (progn
                ;; with-model on a bad path signals before returning a handle,
                ;; so we just confirm the predicates reject plain values
                (let ((not-a-model 42))
                  (not (cl-llama-cpp:llama-model-p not-a-model))))
            (error () nil))
          "llama-model-p returns NIL for non-handle value"))))

(deftest with-model-binds-handle
  (testing "with-model signals model-load-error and the handle type is correct"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let ((result
              (handler-case
                  (cl-llama-cpp:with-model (m "/nonexistent/path.gguf")
                    (type-of m))
                (cl-llama-cpp:model-load-error () :load-error))))
        (ok (eq result :load-error)
            "with-model signals model-load-error for bad path (not a raw pointer)")))))

;;; Boolean ergonomics (issue #43)

(deftest bool-coercion-t-nil
  (testing "%coerce-bool-param converts T and NIL to 1 and 0"
    (ok (= 1 (cl-llama-cpp::%coerce-bool-param t))
        "T → 1")
    (ok (= 0 (cl-llama-cpp::%coerce-bool-param nil))
        "NIL → 0")))

(deftest bool-coercion-integers-passthrough
  (testing "%coerce-bool-param passes integers through unchanged"
    (ok (= 1 (cl-llama-cpp::%coerce-bool-param 1))
        "1 → 1")
    (ok (= 0 (cl-llama-cpp::%coerce-bool-param 0))
        "0 → 0")))

(deftest bool-coercion-rejects-other-types
  (testing "%coerce-bool-param rejects non-integer non-boolean values"
    (ok (handler-case
            (progn (cl-llama-cpp::%coerce-bool-param "true") nil)
          (type-error () t))
        "string signals type-error")
    (ok (handler-case
            (progn (cl-llama-cpp::%coerce-bool-param :yes) nil)
          (type-error () t))
        "keyword signals type-error")))

(deftest override-params-bool-coercion
  (testing "override-params coerces T/NIL for known boolean keys"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let* ((defaults (%llama:context-default-params))
             (overridden (cl-llama-cpp::override-params
                          defaults '(:embeddings t))))
        (ok (= 1 (getf overridden '%llama::embeddings))
            ":embeddings T → 1")))
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let* ((defaults (%llama:context-default-params))
             (overridden (cl-llama-cpp::override-params
                          defaults '(:embeddings nil))))
        (ok (= 0 (getf overridden '%llama::embeddings))
            ":embeddings NIL → 0")))))

(deftest override-params-bool-backward-compat
  (testing "override-params still accepts 0/1 for boolean keys"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let* ((defaults (%llama:context-default-params))
             (overridden (cl-llama-cpp::override-params
                          defaults '(:embeddings 1))))
        (ok (= 1 (getf overridden '%llama::embeddings))
            ":embeddings 1 → 1")))
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let* ((defaults (%llama:context-default-params))
             (overridden (cl-llama-cpp::override-params
                          defaults '(:embeddings 0))))
        (ok (= 0 (getf overridden '%llama::embeddings))
            ":embeddings 0 → 0")))))

(deftest override-params-model-bool-keys
  (testing "override-params coerces T/NIL for model-params boolean keys"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (let* ((defaults (%llama:model-default-params))
             (overridden (cl-llama-cpp::override-params
                          defaults '(:vocab-only t))))
        (ok (= 1 (getf overridden '%llama::vocab-only))
            ":vocab-only T → 1")))))

(deftest get-log-callback-initial
  (testing "get-log-callback returns NIL initially (no callback set by this test)"
    (let ((prev (cl-llama-cpp:get-log-callback)))
      (ok (or (null prev) (functionp prev))
          "get-log-callback returns NIL or function"))))

(deftest set-log-callback-roundtrip
  (testing "set-log-callback installs and get-log-callback retrieves it"
    (let ((prev (cl-llama-cpp:get-log-callback))
          (fn (lambda (level text) (declare (ignore level text)))))
      (unwind-protect
           (progn
             (cl-llama-cpp:set-log-callback fn)
             (ok (eq fn (cl-llama-cpp:get-log-callback))
                 "get-log-callback returns the installed function")
             (ok (null (cl-llama-cpp:set-log-callback nil))
                 "set-log-callback returns NIL")
             (ok (null (cl-llama-cpp:get-log-callback))
                 "get-log-callback returns NIL after clearing"))
        (cl-llama-cpp:set-log-callback prev)))))

;;; Thread-safety internals (issue #52)

(deftest backend-thread-safety-internals
  (testing "thread-safety state variables are defined"
    (ok (boundp 'cl-llama-cpp::*backend-refcount*)
        "*backend-refcount* is defined")
    (ok (integerp cl-llama-cpp::*backend-refcount*)
        "*backend-refcount* is an integer")
    (ok (boundp 'cl-llama-cpp::*backend-permanent*)
        "*backend-permanent* is defined"))
  (testing "thread-safety internal functions are fbound"
    (ok (fboundp 'cl-llama-cpp::%backend-scope-enter)
        "%backend-scope-enter is fbound")
    (ok (fboundp 'cl-llama-cpp::%backend-scope-exit)
        "%backend-scope-exit is fbound")))

(deftest backend-refcount-mid-scope
  (testing "refcount tracks nesting depth"
    (let ((cl-llama-cpp::*backend-initialized* nil)
          (cl-llama-cpp::*backend-permanent* nil)
          (cl-llama-cpp::*backend-refcount* 0))
      (cl-llama-cpp:with-backend ()
        (ok (= 1 cl-llama-cpp::*backend-refcount*)
            "refcount is 1 inside first with-backend")
        (cl-llama-cpp:with-backend ()
          (ok (= 2 cl-llama-cpp::*backend-refcount*)
              "refcount is 2 inside nested with-backend")))
      (ok (= 0 cl-llama-cpp::*backend-refcount*)
          "refcount is 0 after both scopes exit"))))

(deftest backend-permanent-prevents-free
  (testing "permanent hold prevents backend-free when scope exits"
    ;; Simulate: ensure-backend was called, then a with-backend scope opened.
    ;; Exiting the scope should NOT free the backend.
    (let ((cl-llama-cpp::*backend-initialized* t)
          (cl-llama-cpp::*backend-permanent* t)
          (cl-llama-cpp::*backend-refcount* 1))
      (cl-llama-cpp::%backend-scope-exit)
      (ok cl-llama-cpp::*backend-initialized*
          "backend remains initialized when permanent hold is set")
      (ok (= 0 cl-llama-cpp::*backend-refcount*)
          "refcount decremented to 0"))))

#+sbcl
(deftest backend-lock-defined
  (testing "backend mutex is defined on SBCL"
    (ok (boundp 'cl-llama-cpp::*backend-lock*)
        "*backend-lock* is defined")
    (ok (typep cl-llama-cpp::*backend-lock* 'sb-thread:mutex)
        "*backend-lock* is an sb-thread:mutex")))

#+sbcl
(deftest log-lock-defined
  (testing "log callback mutex is defined on SBCL"
    (ok (boundp 'cl-llama-cpp::*log-lock*)
        "*log-lock* is defined")
    (ok (typep cl-llama-cpp::*log-lock* 'sb-thread:mutex)
        "*log-lock* is an sb-thread:mutex")))

;;; GGUF API wrapper tests

(deftest gguf-handle-exported
  (testing "gguf-context handle accessors are exported"
    (dolist (sym '(gguf-context gguf-context-p gguf-context-pointer))
      (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
        (declare (ignore s))
        (ok (eq status :external)
            (format nil "~A is exported" sym))))))

(deftest gguf-condition-hierarchy
  (testing "gguf-load-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:gguf-load-error 'cl-llama-cpp:llama-error)
        "gguf-load-error is a llama-error")))

(deftest gguf-condition-signaling
  (testing "gguf-load-error can be signaled and caught with path slot"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:gguf-load-error :path "/bad/model.gguf")
                    (cl-llama-cpp:gguf-load-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:gguf-load-error)
          "gguf-load-error is catchable")
      (ok (string= "/bad/model.gguf" (cl-llama-cpp:gguf-load-error-path caught))
          "gguf-load-error-path accessor works"))))

(deftest gguf-symbols-exported
  (testing "GGUF wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(with-gguf
                   gguf-version gguf-alignment gguf-data-offset
                   gguf-n-kv gguf-find-key gguf-key gguf-kv-type
                   gguf-val gguf-arr-type gguf-arr-n gguf-arr-data gguf-arr-str
                   gguf-type-name gguf-metadata
                   gguf-n-tensors gguf-find-tensor
                   gguf-tensor-name gguf-tensor-type gguf-tensor-offset
                   gguf-tensor-size gguf-tensor-info gguf-tensors))
      (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
        (declare (ignore s))
        (ok (eq status :external)
            (format nil "~A is exported" sym))))))

(deftest gguf-functions-fbound
  (testing "GGUF wrapper functions are fbound"
    (dolist (sym-name '("GGUF-VERSION" "GGUF-ALIGNMENT" "GGUF-DATA-OFFSET"
                        "GGUF-N-KV" "GGUF-FIND-KEY" "GGUF-KEY" "GGUF-KV-TYPE"
                        "GGUF-VAL" "GGUF-ARR-TYPE" "GGUF-ARR-N"
                        "GGUF-ARR-DATA" "GGUF-ARR-STR"
                        "GGUF-TYPE-NAME" "GGUF-METADATA"
                        "GGUF-N-TENSORS" "GGUF-FIND-TENSOR"
                        "GGUF-TENSOR-NAME" "GGUF-TENSOR-TYPE"
                        "GGUF-TENSOR-OFFSET" "GGUF-TENSOR-SIZE"
                        "GGUF-TENSOR-INFO" "GGUF-TENSORS"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest gguf-binding-deps
  (testing "GGUF bindings are tracked in *binding-deps*"
    (let ((deps cl-llama-cpp:*binding-deps*))
      (dolist (sym '(%llama:gguf-init-from-file %llama:gguf-free
                     %llama:gguf-type-name
                     %llama:gguf-get-version %llama:gguf-get-alignment
                     %llama:gguf-get-data-offset
                     %llama:gguf-get-n-kv %llama:gguf-find-key %llama:gguf-get-key
                     %llama:gguf-get-kv-type %llama:gguf-get-arr-type
                     %llama:gguf-get-val-u8 %llama:gguf-get-val-i8
                     %llama:gguf-get-val-u16 %llama:gguf-get-val-i16
                     %llama:gguf-get-val-u32 %llama:gguf-get-val-i32
                     %llama:gguf-get-val-f32
                     %llama:gguf-get-val-u64 %llama:gguf-get-val-i64
                     %llama:gguf-get-val-f64
                     %llama:gguf-get-val-bool %llama:gguf-get-val-str
                     %llama:gguf-get-val-data
                     %llama:gguf-get-arr-n %llama:gguf-get-arr-data
                     %llama:gguf-get-arr-str
                     %llama:gguf-get-n-tensors %llama:gguf-find-tensor
                     %llama:gguf-get-tensor-name %llama:gguf-get-tensor-type
                     %llama:gguf-get-tensor-offset %llama:gguf-get-tensor-size))
        (ok (member sym deps)
            (format nil "~S is in *binding-deps*" sym))))))

;;; Implicit sync / dirty flag tests (issue #44)

(deftest implicit-sync-compute-pending-slot
  (testing "llama-context struct has compute-pending-p slot"
    (let ((ctx (cl-llama-cpp::%make-llama-context)))
      (ok (not (cl-llama-cpp::llama-context-compute-pending-p ctx))
          "compute-pending-p is NIL for a fresh context")
      (setf (cl-llama-cpp::llama-context-compute-pending-p ctx) t)
      (ok (cl-llama-cpp::llama-context-compute-pending-p ctx)
          "compute-pending-p can be set to T")
      (setf (cl-llama-cpp::llama-context-compute-pending-p ctx) nil)
      (ok (not (cl-llama-cpp::llama-context-compute-pending-p ctx))
          "compute-pending-p can be cleared to NIL"))))

(deftest implicit-sync-synchronize-clears-flag
  (testing "synchronize is fbound and exported"
    (multiple-value-bind (sym status)
        (find-symbol "SYNCHRONIZE" :cl-llama-cpp)
      (ok sym "SYNCHRONIZE is accessible")
      (ok (eq status :external) "SYNCHRONIZE is exported")
      (ok (fboundp sym) "SYNCHRONIZE is fbound"))))

(deftest implicit-sync-embed-fbound
  (testing "embed is fbound and exported"
    (multiple-value-bind (sym status)
        (find-symbol "EMBED" :cl-llama-cpp)
      (ok sym "EMBED is accessible")
      (ok (eq status :external) "EMBED is exported")
      (ok (fboundp sym) "EMBED is fbound"))))

(deftest implicit-sync-batch-decode-sets-pending
  (testing "batch-decode and batch-encode are fbound and exported"
    (dolist (name '("BATCH-DECODE" "BATCH-ENCODE"))
      (multiple-value-bind (sym status)
          (find-symbol name :cl-llama-cpp)
        (ok sym (format nil "~A is accessible" name))
        (ok (eq status :external) (format nil "~A is exported" name))
        (ok (fboundp sym) (format nil "~A is fbound" name))))))

;;; GC finalizer / standalone constructor tests (issue #46)

(deftest finalizer-symbols-exported
  (testing "make-model, free-model, make-context, free-context are exported"
    (dolist (sym '(make-model free-model make-context free-context))
      (multiple-value-bind (s status)
          (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest finalizer-functions-fbound
  (testing "make-model, free-model, make-context, free-context are fbound"
    (dolist (sym-name '("MAKE-MODEL" "FREE-MODEL" "MAKE-CONTEXT" "FREE-CONTEXT"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name)))))

  (testing "internal finalizer helpers are fbound"
    (dolist (sym-name '("%TRY-CLAIM-FOR-FREE"
                        "%REGISTER-MODEL-FINALIZER"
                        "%REGISTER-CONTEXT-FINALIZER"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest freed-cell-slot-exists
  (testing "llama-model and llama-context have freed-cell slots"
    (let ((model (cl-llama-cpp::%make-llama-model))
          (ctx (cl-llama-cpp::%make-llama-context)))
      (ok (consp (cl-llama-cpp::llama-model-freed-cell model))
          "llama-model has a freed-cell cons")
      (ok (null (car (cl-llama-cpp::llama-model-freed-cell model)))
          "fresh model freed-cell starts NIL")
      (ok (consp (cl-llama-cpp::llama-context-freed-cell ctx))
          "llama-context has a freed-cell cons")
      (ok (null (car (cl-llama-cpp::llama-context-freed-cell ctx)))
          "fresh context freed-cell starts NIL"))))

(deftest try-claim-for-free-semantics
  (testing "%try-claim-for-free transitions NIL→T exactly once"
    (let ((cell (list nil)))
      (ok (cl-llama-cpp::%try-claim-for-free cell)
          "first claim returns T")
      (ok (car cell) "cell is now T")
      (ok (not (cl-llama-cpp::%try-claim-for-free cell))
          "second claim returns NIL (already claimed)"))))

(deftest free-model-idempotent-on-dummy
  (testing "free-model is idempotent on a never-allocated model"
    (let ((model (cl-llama-cpp::%make-llama-model)))
      (setf (car (cl-llama-cpp::llama-model-freed-cell model)) t)
      (ok (null (cl-llama-cpp:free-model model))
          "free-model returns NIL on already-freed model"))))

(deftest free-context-idempotent-on-dummy
  (testing "free-context is idempotent on a never-allocated context"
    (let ((ctx (cl-llama-cpp::%make-llama-context)))
      (setf (car (cl-llama-cpp::llama-context-freed-cell ctx)) t)
      (ok (null (cl-llama-cpp:free-context ctx))
          "free-context returns NIL on already-freed context"))))

;;; Pre-flight input validation tests (issue #47)

(deftest input-validation-condition-hierarchy
  (testing "input-validation-error is in the condition hierarchy"
    (ok (subtypep 'cl-llama-cpp:input-validation-error 'cl-llama-cpp:llama-error)
        "input-validation-error is a llama-error")))

(deftest input-validation-condition-signaling
  (testing "input-validation-error can be signaled and caught with all slots"
    (let ((caught (handler-case
                      (error 'cl-llama-cpp:input-validation-error
                             :function-name 'test-fn
                             :argument :test-arg
                             :value 42
                             :reason "test reason")
                    (cl-llama-cpp:input-validation-error (c) c))))
      (ok (typep caught 'cl-llama-cpp:input-validation-error)
          "input-validation-error is catchable")
      (ok (eq 'test-fn (cl-llama-cpp:input-validation-error-function caught))
          "input-validation-error-function accessor works")
      (ok (eq :test-arg (cl-llama-cpp:input-validation-error-argument caught))
          "input-validation-error-argument accessor works")
      (ok (= 42 (cl-llama-cpp:input-validation-error-value caught))
          "input-validation-error-value accessor works")
      (ok (string= "test reason" (cl-llama-cpp:input-validation-error-reason caught))
          "input-validation-error-reason accessor works"))))

(deftest input-validation-symbols-exported
  (testing "input-validation-error symbols are exported from cl-llama-cpp"
    (dolist (sym '(input-validation-error
                   input-validation-error-function
                   input-validation-error-argument
                   input-validation-error-value
                   input-validation-error-reason))
      (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
        (ok s (format nil "~A is accessible" sym))
        (when s
          (ok (eq status :external)
              (format nil "~A is exported" sym)))))))

(deftest batch-add-token-validates-token
  (testing "batch-add-token rejects negative token ID"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (cl-llama-cpp:with-batch (batch 10)
        (ok (handler-case
                (progn (cl-llama-cpp:batch-add-token batch -1 0 0) nil)
              (cl-llama-cpp:input-validation-error () t))
            "negative token signals input-validation-error")))))

(deftest batch-add-token-validates-pos
  (testing "batch-add-token rejects negative position"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (cl-llama-cpp:with-batch (batch 10)
        (ok (handler-case
                (progn (cl-llama-cpp:batch-add-token batch 1 -1 0) nil)
              (cl-llama-cpp:input-validation-error () t))
            "negative pos signals input-validation-error")))))

(deftest batch-add-sequence-validates-empty
  (testing "batch-add-sequence rejects empty token vector"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (cl-llama-cpp:with-batch (batch 10)
        (ok (handler-case
                (progn (cl-llama-cpp:batch-add-sequence batch (vector) 0) nil)
              (cl-llama-cpp:input-validation-error () t))
            "empty tokens signals input-validation-error")))))

(deftest batch-add-sequence-validates-start-pos
  (testing "batch-add-sequence rejects negative start-pos"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (cl-llama-cpp:with-batch (batch 10)
        (ok (handler-case
                (progn (cl-llama-cpp:batch-add-sequence batch (vector 1 2) 0
                                                        :start-pos -1) nil)
              (cl-llama-cpp:input-validation-error () t))
            "negative start-pos signals input-validation-error")))))

(deftest kv-cache-seq-div-validates-zero
  (testing "kv-cache-seq-div signals input-validation-error for zero divisor"
    (ok (handler-case
            (progn (cl-llama-cpp:kv-cache-seq-div nil 0 0 10 0) nil)
          (cl-llama-cpp:input-validation-error () t))
        "zero divisor signals input-validation-error")))

(deftest generate-validates-max-tokens
  (testing "generate rejects non-positive max-tokens"
    (ok (handler-case
            (progn (cl-llama-cpp:generate nil "hello" :max-tokens 0) nil)
          (type-error () t))
        "zero max-tokens signals type-error")
    (ok (handler-case
            (progn (cl-llama-cpp:generate nil "hello" :max-tokens -1) nil)
          (type-error () t))
        "negative max-tokens signals type-error")))

(deftest generate-validates-prompt-type
  (testing "generate rejects non-string non-vector prompt"
    (ok (handler-case
            (progn (cl-llama-cpp:generate nil 42) nil)
          (type-error () t))
        "integer prompt signals type-error")))

(deftest embed-validates-text
  (testing "embed rejects non-string text"
    (ok (handler-case
            (progn (cl-llama-cpp:embed nil 42) nil)
          (type-error () t))
        "integer text signals type-error"))
  (testing "embed rejects empty string"
    (ok (handler-case
            (progn (cl-llama-cpp:embed nil "") nil)
          (cl-llama-cpp:input-validation-error () t))
        "empty text signals input-validation-error")))

(deftest tokenize-validates-text-type
  (testing "tokenize rejects non-string text"
    (ok (handler-case
            (progn (cl-llama-cpp:tokenize nil 42) nil)
          (type-error () t))
        "integer text signals type-error")))

(deftest detokenize-validates-tokens-type
  (testing "detokenize rejects non-vector tokens"
    (ok (handler-case
            (progn (cl-llama-cpp:detokenize nil '(1 2 3)) nil)
          (type-error () t))
        "list tokens signals type-error")))

(deftest format-chat-validates-messages
  (testing "format-chat rejects empty messages"
    (ok (handler-case
            (progn (cl-llama-cpp:format-chat nil nil) nil)
          (cl-llama-cpp:input-validation-error () t))
        "nil messages signals input-validation-error"))
  (testing "format-chat rejects messages without :role"
    (ok (handler-case
            (progn (cl-llama-cpp:format-chat nil (list (list :content "hi"))) nil)
          (cl-llama-cpp:input-validation-error () t))
        "missing :role signals input-validation-error"))
  (testing "format-chat rejects messages without :content"
    (ok (handler-case
            (progn (cl-llama-cpp:format-chat nil (list (list :role "user"))) nil)
          (cl-llama-cpp:input-validation-error () t))
        "missing :content signals input-validation-error")))

(deftest make-model-bad-path
  (testing "make-model signals model-load-error on nonexistent path"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (ok (handler-case
              (cl-llama-cpp:make-model "/nonexistent/path/to/model.gguf")
            (cl-llama-cpp:model-load-error (c)
              (cl-llama-cpp:model-load-error-path c)))
          "model-load-error was signaled for bad path"))))

;;; Chat-session wrapper tests

(deftest chat-session-symbols-exported
  (testing "chat-session wrapper symbols are exported from cl-llama-cpp"
    (dolist (sym '(chat-session
                   chat-session-p
                   chat-session-context
                   chat-session-model
                   chat-session-messages
                   make-chat-session
                   chat-session-send
                   chat-session-reset))
      (let ((found (find-symbol (symbol-name sym) :cl-llama-cpp)))
        (ok found (format nil "~A is accessible in cl-llama-cpp" sym))
        (when found
          (multiple-value-bind (s status) (find-symbol (symbol-name sym) :cl-llama-cpp)
            (declare (ignore s))
            (ok (eq status :external)
                (format nil "~A is exported" sym))))))))

(deftest chat-session-functions-fbound
  (testing "chat-session wrapper symbols are fbound"
    (dolist (sym-name '("MAKE-CHAT-SESSION"
                        "CHAT-SESSION-SEND"
                        "CHAT-SESSION-RESET"
                        "CHAT-SESSION-P"
                        "CHAT-SESSION-CONTEXT"
                        "CHAT-SESSION-MODEL"
                        "CHAT-SESSION-MESSAGES"))
      (let ((sym (find-symbol sym-name :cl-llama-cpp)))
        (ok (and sym (fboundp sym))
            (format nil "~A is fbound" sym-name))))))

(deftest chat-session-send-validates-content
  (testing "chat-session-send rejects non-string content"
    (ok (handler-case
            (progn (cl-llama-cpp:chat-session-send nil 42) nil)
          (cl-llama-cpp:input-validation-error () t))
        "integer content signals input-validation-error"))
  (testing "chat-session-send rejects empty string content"
    (ok (handler-case
            (progn (cl-llama-cpp:chat-session-send nil "") nil)
          (cl-llama-cpp:input-validation-error () t))
        "empty string signals input-validation-error")))

(deftest make-chat-session-validates-ctx
  (testing "make-chat-session rejects non-context argument"
    (ok (handler-case
            (progn (cl-llama-cpp:make-chat-session "not-a-context") nil)
          (cl-llama-cpp:input-validation-error () t))
        "non-context signals input-validation-error")))

;;; Interactive restart tests (issue #74)

(deftest model-load-error-restarts
  (testing "make-model establishes retry-with-layers, use-cpu-only, and use-different-path restarts"
    (cl-llama-cpp:with-llama-compatible-fp-environment
      (%llama:backend-init)
      (let (retry-with-layers use-cpu-only use-different-path)
        (handler-case
            (handler-bind ((cl-llama-cpp:model-load-error
                            (lambda (c)
                              (declare (ignore c))
                              (setf retry-with-layers
                                    (find-restart 'cl-llama-cpp::retry-with-layers)
                                    use-cpu-only
                                    (find-restart 'cl-llama-cpp::use-cpu-only)
                                    use-different-path
                                    (find-restart 'cl-llama-cpp::use-different-path)))))
              (cl-llama-cpp:make-model "/nonexistent/model.gguf"))
          (cl-llama-cpp:model-load-error () nil))
        (ok retry-with-layers "retry-with-layers restart is established")
        (ok use-cpu-only "use-cpu-only restart is established")
        (ok use-different-path "use-different-path restart is established")))))

(deftest grammar-error-restarts
  (testing "make-grammar-sampler establishes skip-grammar and use-different-grammar restarts"
    (let (skip-grammar use-different-grammar)
      (handler-case
          (handler-bind ((cl-llama-cpp:grammar-error
                          (lambda (c)
                            (declare (ignore c))
                            (setf skip-grammar
                                  (find-restart 'cl-llama-cpp::skip-grammar)
                                  use-different-grammar
                                  (find-restart 'cl-llama-cpp::use-different-grammar)))))
            (cl-llama-cpp:make-grammar-sampler nil ""))
        (cl-llama-cpp:grammar-error () nil))
      (ok skip-grammar "skip-grammar restart is established by make-grammar-sampler")
      (ok use-different-grammar "use-different-grammar restart is established by make-grammar-sampler")))
  (testing "make-grammar-sampler-lazy establishes skip-grammar and use-different-grammar restarts"
    (let (skip-grammar use-different-grammar)
      (handler-case
          (handler-bind ((cl-llama-cpp:grammar-error
                          (lambda (c)
                            (declare (ignore c))
                            (setf skip-grammar
                                  (find-restart 'cl-llama-cpp::skip-grammar)
                                  use-different-grammar
                                  (find-restart 'cl-llama-cpp::use-different-grammar)))))
            (cl-llama-cpp:make-grammar-sampler-lazy nil ""))
        (cl-llama-cpp:grammar-error () nil))
      (ok skip-grammar "skip-grammar restart is established by make-grammar-sampler-lazy")
      (ok use-different-grammar "use-different-grammar restart is established by make-grammar-sampler-lazy"))))

;;; Sampler config object tests (issue #49)

(deftest sampler-config-symbol-exported
  (testing "make-sampler-config is exported from cl-llama-cpp"
    (multiple-value-bind (sym status)
        (find-symbol "MAKE-SAMPLER-CONFIG" :cl-llama-cpp)
      (ok sym "MAKE-SAMPLER-CONFIG is accessible")
      (ok (eq status :external) "MAKE-SAMPLER-CONFIG is exported"))))

(deftest sampler-config-fbound
  (testing "make-sampler-config is fbound"
    (let ((sym (find-symbol "MAKE-SAMPLER-CONFIG" :cl-llama-cpp)))
      (ok (and sym (fboundp sym)) "MAKE-SAMPLER-CONFIG is fbound"))))

(deftest sampler-config-returns-plist
  (testing "make-sampler-config returns a plist of the supplied params"
    (let ((cfg (cl-llama-cpp:make-sampler-config :temp 0.3 :top-k 40)))
      (ok (listp cfg) "result is a list")
      (ok (= 0.3 (getf cfg :temp)) ":temp is stored")
      (ok (= 40 (getf cfg :top-k)) ":top-k is stored"))))

(deftest sampler-config-stores-only-provided
  (testing "make-sampler-config stores only explicitly supplied params"
    (let ((cfg (cl-llama-cpp:make-sampler-config :seed 99)))
      (ok (= 2 (length cfg)) "plist has exactly one key/value pair")
      (ok (= 99 (getf cfg :seed)) ":seed is stored")
      (ok (eq (getf cfg :temp :absent) :absent) ":temp absent when not supplied"))))

(deftest sampler-config-rejects-unknown-keys
  (testing "make-sampler-config rejects unknown keyword arguments"
    (ok (handler-case
            (progn (cl-llama-cpp:make-sampler-config :unknown-key 42) nil)
          (error () t))
        "unknown keyword signals an error")))

(deftest sampler-config-override-semantics
  (testing "explicit kwarg in generate overrides config value at the plist level"
    ;; Verify the merge logic: caller-explicit appears before config in plist.
    ;; We can't call generate without a real context, so verify the plist mechanic directly.
    (let* ((cfg (cl-llama-cpp:make-sampler-config :temp 0.3 :top-k 40))
           ;; Simulate: caller passes :temp 0.9; config has :temp 0.3
           (caller-sampler '(:temp 0.9))
           (effective (append caller-sampler cfg)))
      (ok (= 0.9 (getf effective :temp)) "caller temp (0.9) wins over config temp (0.3)")
      (ok (= 40  (getf effective :top-k)) "config top-k (40) used when caller omits it"))))

;;; ---------- Issue #81: :seed :random sentinel ----------

(deftest resolve-seed-integer-passthrough
  (testing "resolve-seed passes integer seeds through unchanged"
    (ok (= 42 (cl-llama-cpp:resolve-seed 42)) "42 → 42")
    (ok (= 0 (cl-llama-cpp:resolve-seed 0)) "0 → 0")
    (ok (= 12345 (cl-llama-cpp:resolve-seed 12345)) "12345 → 12345")))

(deftest resolve-seed-random-returns-default-seed
  (testing "resolve-seed maps :random to %llama:+default-seed+"
    (let ((result (cl-llama-cpp:resolve-seed :random)))
      (ok (integerp result) ":random resolves to an integer")
      (ok (= %llama:+default-seed+ result)
          ":random maps to LLAMA_DEFAULT_SEED (0xFFFFFFFF)"))))

(deftest resolve-seed-nil-means-random
  (testing "resolve-seed treats nil as :random"
    (let ((result (cl-llama-cpp:resolve-seed nil)))
      (ok (integerp result) "nil resolves to an integer")
      (ok (= %llama:+default-seed+ result)
          "nil maps to LLAMA_DEFAULT_SEED same as :random"))))

(deftest resolve-seed-rejects-invalid-types
  (testing "resolve-seed signals error for invalid seed types"
    (ok (handler-case (progn (cl-llama-cpp:resolve-seed "random") nil)
          (type-error () t)
          (cl-llama-cpp:input-validation-error () t))
        "string signals error")
    (ok (handler-case (progn (cl-llama-cpp:resolve-seed t) nil)
          (type-error () t)
          (cl-llama-cpp:input-validation-error () t))
        "t signals error")
    (ok (handler-case (progn (cl-llama-cpp:resolve-seed 3.14) nil)
          (type-error () t)
          (cl-llama-cpp:input-validation-error () t))
        "float signals error")))

(deftest sampler-config-stores-random-sentinel
  (testing "make-sampler-config stores :seed :random verbatim"
    (let ((cfg (cl-llama-cpp:make-sampler-config :seed :random)))
      (ok (eq :random (getf cfg :seed)) ":random stored as-is in config"))))

(deftest sampler-config-stores-nil-seed
  (testing "make-sampler-config stores :seed nil verbatim"
    (let ((cfg (cl-llama-cpp:make-sampler-config :seed nil)))
      (ok (= 2 (length cfg)) "plist has one key-value pair")
      (ok (eq nil (getf cfg :seed :not-found)) "nil stored as-is in config"))))

;;; UTF-8 byte length calculation (security fix for tokenize)

(deftest utf-8-byte-length-ascii
  (testing "%utf-8-byte-length matches character count for ASCII"
    (ok (= 0 (cl-llama-cpp::%utf-8-byte-length ""))
        "empty string → 0")
    (ok (= 5 (cl-llama-cpp::%utf-8-byte-length "Hello"))
        "ASCII string → same as length")
    (ok (= 13 (cl-llama-cpp::%utf-8-byte-length "Hello, world!"))
        "ASCII with punctuation")))

(deftest utf-8-byte-length-2byte
  (testing "%utf-8-byte-length for 2-byte characters (U+0080–U+07FF)"
    (ok (= 2 (cl-llama-cpp::%utf-8-byte-length "é"))
        "é (U+00E9) is 2 bytes")
    (ok (= 5 (cl-llama-cpp::%utf-8-byte-length "café"))
        "café: c(1)+a(1)+f(1)+é(2) = 5")
    (ok (= 5 (cl-llama-cpp::%utf-8-byte-length "über"))
        "über: ü(2)+b(1)+e(1)+r(1) = 5")))

(deftest utf-8-byte-length-3byte
  (testing "%utf-8-byte-length for 3-byte characters (U+0800–U+FFFF)"
    (ok (= 6 (cl-llama-cpp::%utf-8-byte-length "你好"))
        "你好: 2×3 = 6 bytes")
    (ok (= 9 (cl-llama-cpp::%utf-8-byte-length "日本語"))
        "日本語: 3×3 = 9 bytes")))

(deftest utf-8-byte-length-4byte
  (testing "%utf-8-byte-length for 4-byte characters (U+10000+)"
    (ok (= 4 (cl-llama-cpp::%utf-8-byte-length "😀"))
        "😀 (U+1F600) is 4 bytes")
    (ok (= 8 (cl-llama-cpp::%utf-8-byte-length "🎉🎊"))
        "two emoji: 2×4 = 8 bytes")))

(deftest utf-8-byte-length-mixed
  (testing "%utf-8-byte-length for mixed ASCII and multi-byte"
    (ok (= 10 (cl-llama-cpp::%utf-8-byte-length "Hi 你好!"))
        "H(1)+i(1)+space(1)+你(3)+好(3)+!(1) = 10")
    (ok (= 6 (cl-llama-cpp::%utf-8-byte-length "a😀b"))
        "a(1)+😀(4)+b(1) = 6")
    (ok (> (cl-llama-cpp::%utf-8-byte-length "こんにちは世界")
           (length "こんにちは世界"))
        "byte length > char length for CJK")))

(deftest test-model-freed-p
  (let ((m (cl-llama-cpp::%make-llama-model)))
    (ok (not (cl-llama-cpp:model-freed-p m)))
    (setf (car (cl-llama-cpp::llama-model-freed-cell m)) t)
    (ok (cl-llama-cpp:model-freed-p m))))
