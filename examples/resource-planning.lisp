;;; Resource planning & configuration validation example — estimates
;;; memory requirements, validates configurations, and suggests
;;; alternatives when a budget is too tight.
;;;
;;; Setup:
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/resource-planning.lisp")
;;;   (setf cl-llama-cpp/examples/resource-planning::*model-path*
;;;         "/path/to/model.gguf")
;;;   (cl-llama-cpp/examples/resource-planning:run)
;;;
;;; Or via environment variable:
;;;   export LLAMA_MODEL=/path/to/model.gguf
;;;
;;; All sections share a single model load. The runtime-guardrails demo
;;; creates real contexts in addition to the loaded model — it is OFF by
;;; default. Enable it with:
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

(defun format-bytes (bytes)
  (cond
    ((>= bytes (* 1024 1024 1024))
     (format nil "~,1F GiB" (/ bytes (* 1024.0d0 1024 1024))))
    ((>= bytes (* 1024 1024))
     (format nil "~,1F MiB" (/ bytes (* 1024.0d0 1024))))
    (t (format nil "~,1F KiB" (/ bytes 1024.0d0)))))

;;; ── Main ─────────────────────────────────────────────────────────────

(defun run (&key (vram-gib 8) guardrails)
  "Run the resource planning demo.
VRAM-GIB is the simulated VRAM budget in GiB (default 8) used for the
explicit-budget sections. When GUARDRAILS is non-nil, also demonstrate
the runtime :validation hooks of WITH-CONTEXT by creating real contexts;
this is disabled by default."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (let* ((vram-budget   (ceiling (* vram-gib 1024 1024 1024)))
         ;; Snapshot free VRAM before loading so we plan against the full
         ;; available budget, not the reduced headroom after weights load.
         (pre-load-free (detect-free-vram))
         (pre-load-total (detect-total-vram))
         (gpu-devs      (gpu-devices)))

    ;; ── GPU detection ────────────────────────────────────────────────────
    (banner "Hardware Detection")
    (if gpu-devs
        (progn
          (format t "  GPU device~P detected:~%" (length gpu-devs))
          (dolist (d gpu-devs)
            (format t "    ~A  (~A)~%"
                    (getf d :name)
                    (getf d :description)))
          (format t "~%  Free VRAM:  ~A  (snapshot before model load)~%"
                  (format-bytes pre-load-free))
          (format t "  Total VRAM: ~A~%" (format-bytes pre-load-total)))
        (format t "  No GPU detected — CPU-only build.~%"))

    (format t "~&Loading model: ~A~%" *model-path*)

    (with-model (model *model-path*)

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

      ;; ── Feasibility report — pre-load free VRAM ──────────────────────
      ;; We pass the snapshot taken before loading so the budget reflects
      ;; total available VRAM, not the reduced headroom after weights load.
      (banner "Feasibility Report (detected VRAM, pre-load)")
      (format t "  Default context, all layers on GPU:~%")
      (feasibility-report model :vram-budget pre-load-free)

      (section "Large context, all layers on GPU")
      (feasibility-report model :n-ctx 32768 :vram-budget pre-load-free)

      ;; ── Feasibility report — explicit budget ─────────────────────────
      (banner (format nil "Feasibility Report (explicit budget: ~D GiB)" vram-gib))

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
        (format t "  n-ctx=4096,  explicit ~D GiB:       ~S~%" vram-gib result))

      (let ((result (validate-configuration model :n-ctx 32768
                                                  :vram-budget vram-budget)))
        (format t "  n-ctx=32768, explicit ~D GiB:       ~S~%" vram-gib result))

      (let ((result (validate-configuration model :n-ctx 4096
                                                  :vram-budget pre-load-free)))
        (format t "  n-ctx=4096,  detected (pre-load):  ~S~%" result))

      ;; ── Configuration suggestions ────────────────────────────────────
      (banner (format nil "Configuration Suggestions (explicit budget: ~D GiB)" vram-gib))

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
            (format t "~%  No viable configuration found for n-ctx=8192.~%")))

      (banner "Configuration Suggestions (detected VRAM, pre-load)")

      (let ((suggestion (suggest-configuration model :n-ctx 32768
                                                     :n-gpu-layers 999
                                                     :vram-budget pre-load-free)))
        (if suggestion
            (format t "  For n-ctx=32768, n-gpu-layers=999:~%    Suggested: ~S~%"
                    suggestion)
            (format t "  No viable configuration found (no GPU or budget too tight).~%")))

      (let ((suggestion (suggest-configuration model :n-ctx 8192
                                                     :vram-budget pre-load-free)))
        (if suggestion
            (format t "~%  For n-ctx=8192:~%    Suggested: ~S~%" suggestion)
            (format t "~%  No viable configuration found (no GPU or budget too tight).~%"))))

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
            (format t "  [ERROR] Caught: ~A~%" c)))

        (format t "~%  Creating context with :validation :warn, auto-detected budget...~%")
        (handler-bind ((configuration-unsafe-warning
                        (lambda (c)
                          (format t "  [WARNING] ~A~%" c)
                          (muffle-warning c))))
          (with-context (ctx model :n-ctx 512 :validation :warn)
            (format t "  Context created (n-ctx=~D)~%"
                    (getf (context-info ctx) :n-ctx)))))))

  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  Demo complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
