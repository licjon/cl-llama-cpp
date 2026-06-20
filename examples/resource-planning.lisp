;;; Resource planning & configuration validation example — estimates
;;; memory requirements, validates configurations, and suggests
;;; alternatives when a budget is too tight.
;;;
;;; Setup:
;;;   export LLAMA_MODEL=/path/to/model.gguf    ; or set *model-path* in the REPL
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/resource-planning.lisp")
;;;   (cl-llama-cpp/examples/resource-planning:run)
;;;
;;; The estimation/validation sections use a lightweight :vocab-only model
;;; load (fast). The runtime-guardrails demo loads the FULL model to create
;;; real contexts, which is slow and CPU-bound — it is OFF by default. Enable
;;; it with:
;;;   (cl-llama-cpp/examples/resource-planning:run :guardrails t)

(defpackage #:cl-llama-cpp/examples/resource-planning
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/resource-planning)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun section (title)
  (format t "~&~%  ── ~A ──~2%" title))

;;; ── Main ─────────────────────────────────────────────────────────────

(defun run (&key (vram-gib 8) guardrails)
  "Run the resource planning demo.
VRAM-GIB is the simulated VRAM budget in GiB (default 8).
When GUARDRAILS is non-nil, also demonstrate the runtime :validation hooks
of WITH-CONTEXT. This loads the FULL model (not vocab-only) and is slow and
CPU-bound, so it is disabled by default."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (let ((vram-budget (ceiling (* vram-gib 1024 1024 1024))))
    (format t "~&Loading model (vocab-only): ~A~%" *model-path*)

    ;; :vocab-only 1 reads just the metadata — no weights, no offload.
    (with-model (model *model-path* :vocab-only 1)

      ;; ── Basic memory estimate ────────────────────────────────────────
      (banner "Memory Estimation")
      (format t "  Using default context (training length) with F16 KV cache:~%")
      (explain-memory-usage model)

      ;; ── Estimate with custom parameters ──────────────────────────────
      (section "Custom Configuration: n-ctx=8192, Q8_0 KV cache")
      (explain-memory-usage model :n-ctx 8192 :type-k :q8-0 :type-v :q8-0)

      ;; ── Estimate with large context ──────────────────────────────────
      (section "Large Context: n-ctx=32768")
      (explain-memory-usage model :n-ctx 32768)

      ;; ── Feasibility report ───────────────────────────────────────────
      (banner (format nil "Feasibility Report (budget: ~D GiB)" vram-gib))

      (format t "  Default context, all layers on GPU:~%")
      (feasibility-report model :vram-budget vram-budget)

      (section "Large context, all layers on GPU")
      (feasibility-report model :n-ctx 32768 :vram-budget vram-budget)

      (section "Large context, 10 GPU layers only")
      (feasibility-report model :n-ctx 32768 :n-gpu-layers 10
                                :vram-budget vram-budget)

      ;; ── Programmatic validation ──────────────────────────────────────
      (banner "Programmatic Validation")

      (let ((result (validate-configuration model :n-ctx 4096
                                                  :vram-budget vram-budget)))
        (format t "  n-ctx=4096:  ~S~%" result))

      (let ((result (validate-configuration model :n-ctx 32768
                                                  :vram-budget vram-budget)))
        (format t "  n-ctx=32768: ~S~%" result))

      (let ((result (validate-configuration model :n-ctx 4096)))
        (format t "  no budget:   ~S~%" result))

      ;; ── Configuration suggestions ────────────────────────────────────
      (banner (format nil "Configuration Suggestions (budget: ~D GiB)" vram-gib))

      (let ((suggestion (suggest-configuration model :n-ctx 32768
                                                     :n-gpu-layers 999
                                                     :vram-budget vram-budget)))
        (if suggestion
            (format t "  For n-ctx=32768, n-gpu-layers=999:~%    Suggested: ~S~%"
                    suggestion)
            (format t "  No viable configuration found for n-ctx=32768.~%")))

      (let ((suggestion (suggest-configuration model :n-ctx 8192
                                                     :vram-budget vram-budget)))
        (if suggestion
            (format t "~%  For n-ctx=8192:~%    Suggested: ~S~%" suggestion)
            (format t "~%  No viable configuration found for n-ctx=8192.~%"))))

    ;; ── Runtime guardrails demo (optional) ────────────────────────────────
    ;; Loads the FULL model to create real contexts. n-gpu-layers is left at
    ;; the model default (offload to GPU when the build supports it); pass
    ;; :guardrails t to run this section.
    (when guardrails
      (banner "Runtime Guardrails")
      (with-model (model *model-path*)
        (format t "  Creating context with :validation :warn and tight budget...~%")
        (handler-bind ((configuration-unsafe-warning
                        (lambda (c)
                          (format t "  [WARNING] ~A~%" c)
                          (muffle-warning c))))
          (with-context (ctx model :n-ctx 512
                                   :validation :warn
                                   :vram-budget 1024)
            (format t "  Context created despite warning (n-ctx=~D)~%"
                    (getf (context-info ctx) :n-ctx))))

        (format t "~%  Creating context with :validation :error and tight budget...~%")
        (handler-case
            (with-context (ctx model :n-ctx 512
                                     :validation :error
                                     :vram-budget 1024)
              ctx)
          (configuration-unsafe-error (c)
            (format t "  [ERROR] Caught: ~A~%" c))))))

  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  Demo complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
