(defsystem "cl-llama-cpp"
  :version "0.2.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :description "CFFI bindings to llama.cpp"
  :depends-on ("cffi" "cffi-libffi" "trivial-garbage" "form-fiddle")
  :serial t
  :components ((:module "src"
                :serial t
                :components
                ((:file "packages")
                 (:file "library")
                 (:file "bindings")
                 (:file "conditions")
                 (:file "handles")
                 (:file "binding-deps")
		 (:file "introspection")
		 (:file "tokenization")
		 (:file "resource-planning")
		 (:file "lifecycle")
		 (:file "chat")
		 (:file "grammar")
		 (:file "batch")
		 (:file "generation")
		 (:file "kv-cache")
		 (:file "session")
		 (:file "chat-session")
		 (:file "lora")
		 (:file "gguf")))))

(defsystem "cl-llama-cpp/examples"
  :description "Example programs for cl-llama-cpp"
  :depends-on ("cl-llama-cpp" "cl-llama-cpp/common/json-schema")
  :components ((:module "examples"
                :components
                ((:file "simple-chat")
                 (:file "incremental-chat")
                 (:file "backend-lifecycle")
		 (:file "benchmark-hotpaths")
                 (:file "context-fork")
                 (:file "introspection")
                 (:file "kv-cache")
                 (:file "lora")
                 (:file "parallel")
                 (:file "parallel-threads")
                 (:file "perf-and-logging")
                 (:file "resource-planning")
                 (:file "sampler-comparison")
                 (:file "sampler-showcase")
                 (:file "tool-calling")
                 (:file "json-schema")))))

(defsystem "cl-llama-cpp/generate"
  :description "Binding generator for cl-llama-cpp (developers only)"
  :depends-on ("claw" "cl-llama-cpp")
  :components ((:module "generate"
                :components
                ((:file "generate")))))

(defsystem "cl-llama-cpp/tests"
  :description "Tests for cl-llama-cpp"
  :depends-on ("cl-llama-cpp" "rove")
  :components ((:module "tests"
                :components
                ((:file "smoke")
                 (:file "integration"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp")
  :components ((:module "common"
                :components
                ((:file "common"))))
  :description "Optional cl-llama-cpp extensions mirroring llama.cpp's common/ layer"
  :in-order-to ((test-op (test-op "cl-llama-cpp/common/tests"))))

(defsystem "cl-llama-cpp/common/shim"
  :description "Shared C++ shim build and load infrastructure for cl-llama-cpp/common"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp")
  :serial t
  :components ((:module "common/shim"
                :serial t
                :components
                ((:file "packages")
                 (:file "library")))))

(defsystem "cl-llama-cpp/common/json-schema"
  :description "JSON Schema to GBNF grammar conversion for cl-llama-cpp"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim" "yason")
  :serial t
  :components ((:module "common/json-schema"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "json-schema")))))

(defsystem "cl-llama-cpp/common/json-schema/tests"
  :description "Tests for cl-llama-cpp/common/json-schema"
  :depends-on ("cl-llama-cpp/common/json-schema" "rove")
  :components ((:module "tests"
                :components
                ((:file "json-schema"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/json-partial"
  :description "Streaming JSON healer — parse incomplete JSON mid-generation"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim" "yason")
  :serial t
  :components ((:module "common/json-partial"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "json-partial")))))

(defsystem "cl-llama-cpp/common/json-partial/tests"
  :description "Tests for cl-llama-cpp/common/json-partial"
  :depends-on ("cl-llama-cpp/common/json-partial" "rove")
  :components ((:module "tests"
                :components
                ((:file "json-partial"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/speculative"
  :description "Speculative decoding for cl-llama-cpp"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim" "trivial-garbage")
  :serial t
  :components ((:module "common/speculative"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "speculative")))))

(defsystem "cl-llama-cpp/common/speculative/tests"
  :description "Tests for cl-llama-cpp/common/speculative"
  :depends-on ("cl-llama-cpp/common/speculative" "rove")
  :components ((:module "tests"
                :components
                ((:file "speculative"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/reasoning-budget"
  :description "Reasoning token budget limiter for cl-llama-cpp"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim")
  :serial t
  :components ((:module "common/reasoning-budget"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "reasoning-budget")))))

(defsystem "cl-llama-cpp/common/reasoning-budget/tests"
  :description "Tests for cl-llama-cpp/common/reasoning-budget"
  :depends-on ("cl-llama-cpp/common/reasoning-budget" "rove")
  :components ((:module "tests"
                :components
                ((:file "reasoning-budget"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/ngram-map"
  :description "N-gram speculative token drafting for cl-llama-cpp"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim" "trivial-garbage")
  :serial t
  :components ((:module "common/ngram-map"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "ngram-map")))))

(defsystem "cl-llama-cpp/common/ngram-map/tests"
  :description "Tests for cl-llama-cpp/common/ngram-map"
  :depends-on ("cl-llama-cpp/common/ngram-map" "rove")
  :components ((:module "tests"
                :components
                ((:file "ngram-map"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/chat"
  :description "Chat template rendering and tool call parsing for cl-llama-cpp"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim" "yason" "trivial-garbage")
  :serial t
  :components ((:module "common/chat"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "chat")))))

(defsystem "cl-llama-cpp/common/chat/tests"
  :description "Tests for cl-llama-cpp/common/chat"
  :depends-on ("cl-llama-cpp/common/chat" "rove")
  :components ((:module "tests"
                :components
                ((:file "chat"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/imatrix-loader"
  :description "Importance matrix loading for quantization workflows"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim")
  :serial t
  :components ((:module "common/imatrix-loader"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "imatrix-loader")))))

(defsystem "cl-llama-cpp/common/imatrix-loader/tests"
  :description "Tests for cl-llama-cpp/common/imatrix-loader"
  :depends-on ("cl-llama-cpp/common/imatrix-loader" "rove")
  :components ((:module "tests"
                :components
                ((:file "imatrix-loader"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/fit"
  :description "Automatic GPU memory fitting for cl-llama-cpp"
  :version "0.1.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common/shim")
  :serial t
  :components ((:module "common/fit"
                :serial t
                :components
                ((:file "packages")
                 (:file "bindings")
                 (:file "fit")))))

(defsystem "cl-llama-cpp/common/fit/tests"
  :description "Tests for cl-llama-cpp/common/fit"
  :depends-on ("cl-llama-cpp/common/fit" "rove")
  :components ((:module "tests"
                :components
                ((:file "fit"))))
  :perform (test-op (op c) (symbol-call :rove :run c)))

(defsystem "cl-llama-cpp/common/tests"
  :author "Jonathan Hustad"
  :license "MIT"
  :depends-on ("cl-llama-cpp/common"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "common"))))
  :description "Test system for cl-llama-cpp/common"
  :perform (test-op (op c) (symbol-call :rove :run c)))
