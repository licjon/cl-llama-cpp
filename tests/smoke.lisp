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
    ;; Rebind to guarantee clean state regardless of prior tests.
    (let ((cl-llama-cpp::*backend-initialized* nil)
          (init-count 0))
      (cl-llama-cpp:with-backend ()
        (incf init-count)
        (cl-llama-cpp:with-backend ()
          (incf init-count)))
      (ok (= 2 init-count) "body ran twice (nesting works)")
      (ok (not cl-llama-cpp::*backend-initialized*)
          "*backend-initialized* cleared after outermost exit"))))

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
