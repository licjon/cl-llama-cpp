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
    (cl-llama-cpp:with-fp-traps-masked
      (let ((params (%llama:model-default-params)))
        (ok params "model-default-params returned non-nil")
        (ok (listp params) "result is a plist")))))

(deftest context-default-params
  (testing "llama_context_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-fp-traps-masked
      (let ((params (%llama:context-default-params)))
        (ok params "context-default-params returned non-nil")
        (ok (listp params) "result is a plist")))))

(deftest sampler-chain-default-params
  (testing "llama_sampler_chain_default_params returns a struct via cffi-libffi"
    (cl-llama-cpp:with-fp-traps-masked
      (let ((params (%llama:sampler-chain-default-params)))
        (ok (listp params) "result is a plist")))))

(deftest backend-init
  (testing "llama_backend_init runs without error"
    (cl-llama-cpp:with-fp-traps-masked
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
    (cl-llama-cpp:with-fp-traps-masked
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
