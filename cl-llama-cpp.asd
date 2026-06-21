(defsystem "cl-llama-cpp"
  :version "0.2.0"
  :author "Jonathan Hustad"
  :license "MIT"
  :description "CFFI bindings to llama.cpp"
  :depends-on ("cffi" "cffi-libffi")
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
		 (:file "generation")
		 (:file "batch")
		 (:file "kv-cache")
		 (:file "session")
		 (:file "lora")))))

(defsystem "cl-llama-cpp/examples"
  :description "Example programs for cl-llama-cpp"
  :depends-on ("cl-llama-cpp")
  :components ((:module "examples"
                :components
                ((:file "simple-chat")
                 (:file "backend-lifecycle")
                 (:file "context-fork")
                 (:file "introspection")
                 (:file "kv-cache")
                 (:file "lora")
                 (:file "parallel")
                 (:file "perf-and-logging")
                 (:file "resource-planning")
                 (:file "sampler-comparison")
                 (:file "sampler-showcase")))))

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
