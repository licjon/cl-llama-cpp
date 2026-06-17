;;; Performance counters, logging, and system info example.
;;;
;;; system-capabilities  — query mmap/mlock/GPU/RPC support and device count
;;; time-us              — microsecond wall-clock timestamp
;;; set-log-callback     — redirect llama.cpp log output to a Lisp function
;;; get-log-callback     — retrieve the current Lisp callback
;;; context-perf         — structured timing data after inference
;;; sampler-perf         — structured timing data for the sampler chain
;;; print-perf           — print context perf to stderr
;;; reset-perf           — zero all context perf counters
;;; with-perf            — reset before body, print after (even on error)
;;;
;;; Setup:
;;;   export LLAMA_MODEL=~/models/gemma-3-1b-it-Q4_K_M.gguf
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/perf-and-logging.lisp")
;;;   (cl-llama-cpp/examples/perf-and-logging:run)

(defpackage #:cl-llama-cpp/examples/perf-and-logging
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/perf-and-logging)

(defparameter *model-path*
  (or (uiop:getenv "LLAMA_MODEL")
      (error "Set LLAMA_MODEL to the path of a GGUF chat model.")))


;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun log-level-name (level)
  (case level
    (1 "DEBUG") (2 "INFO") (3 "WARN") (4 "ERROR") (5 "CONT") (t "???")))

;;; ── Main ─────────────────────────────────────────────────────────────

(defun run ()
  "Run all performance / logging demonstrations."

  ;; ════════════════════════════════════════════════════════════════════
  (banner "PHASE 1: System capabilities (no model required)")
  ;; ════════════════════════════════════════════════════════════════════
  ;;
  ;; system-capabilities queries five C functions and returns a single
  ;; plist. The :supports-* values are booleans; :max-devices is an
  ;; integer. These are build-time and hardware constants — they never
  ;; change for a given binary.

  (let ((caps (system-capabilities)))
    (format t "  :mmap        ~A~%" (getf caps :mmap))
    (format t "  :mlock       ~A~%" (getf caps :mlock))
    (format t "  :gpu-offload ~A~%" (getf caps :gpu-offload))
    (format t "  :rpc         ~A~%" (getf caps :rpc))
    (format t "  :max-devices ~A~%" (getf caps :max-devices)))

  (let ((t0 (time-us))
        (t1 (time-us)))
    (format t "~%  time-us sample: ~D µs  (two calls differ by ~D µs)~%"
            t0 (- t1 t0)))

  (format t "~%  system-info string (first 120 chars):~%  ~A~%"
          (let ((s (system-info))) (subseq s 0 (min 120 (length s)))))

  (format t "~%✓ System capabilities queried without loading a model.~%")

  (with-backend ()

    ;; ════════════════════════════════════════════════════════════════
    (banner "PHASE 2: Redirecting log output to Lisp")
    ;; ════════════════════════════════════════════════════════════════
    ;;
    ;; By default llama.cpp writes log messages to stderr.
    ;; set-log-callback replaces that with any Lisp function, letting
    ;; you filter, count, or collect messages during model load.
    ;; Passing NIL restores the default stderr logger.

    (let ((counts (list 1 0  2 0  3 0  4 0))   ; level → message count
          (samples '()))
      (set-log-callback
       (lambda (level text)
         (incf (getf counts level 0))
         (when (< (length samples) 3)
           (push (list level text) samples))))

      (format t "  Loading model with log capture active...~%")
      (with-model (model *model-path* :n-gpu-layers 0)
        (set-log-callback nil)   ; restore before any output
        (format t "  Log message counts during load:~%")
        (loop for (level count) on counts by #'cddr
              when (plusp count)
              do (format t "    ~5A ~D message~:P~%"
                         (log-level-name level) count))
        (format t "~%  First few captured messages:~%")
        (dolist (entry (nreverse samples))
          (format t "    [~A] ~A"
                  (log-level-name (first entry))
                  (string-right-trim '(#\newline #\return) (second entry)))
          (terpri))
        (format t "~%✓ Log callback installed, messages captured, callback cleared.~%")

        ;; ════════════════════════════════════════════════════════════
        (banner "PHASE 3: with-perf macro")
        ;; ════════════════════════════════════════════════════════════
        ;;
        ;; with-perf resets context perf counters before the body runs,
        ;; then calls print-context-perf (→ stderr) after it completes
        ;; — even on non-local exit. The macro returns the body's values.

        (with-context (ctx model :n-ctx 512)

          (format t "  Running generation inside (with-perf (ctx) ...):~%")
          (format t "  (perf output goes to stderr)~%~%")

          (let ((text (with-perf (ctx)
                        (generate ctx "The speed of light is"
                                  :max-tokens 32 :temp 0.1))))
            (format t "~%  Generated: ~S~%" text))

          (format t "~%✓ with-perf reset counters and printed timing on exit.~%")

          ;; ══════════════════════════════════════════════════════════
          (banner "PHASE 4: Structured perf data and throughput")
          ;; ══════════════════════════════════════════════════════════
          ;;
          ;; context-perf returns the same numbers print-perf writes to
          ;; stderr, but as a Lisp plist you can compute with.
          ;; sampler-perf similarly exposes sampling time and call count.

          (reset-perf ctx)
          (let ((t-before (time-us)))
            (generate ctx "Roses are red," :max-tokens 32 :temp 0.1)
            (let* ((t-after  (time-us))
                   (elapsed-ms (/ (- t-after t-before) 1000.0))
                   (cp (context-perf ctx)))

              (format t "  context-perf plist:~%")
              (format t "    :t-p-eval-ms  ~,2F ms  (~D prompt tokens)~%"
                      (getf cp :t-p-eval-ms) (getf cp :n-p-eval))
              (format t "    :t-eval-ms    ~,2F ms  (~D generated tokens)~%"
                      (getf cp :t-eval-ms) (getf cp :n-eval))
              (format t "    :n-reused     ~D (KV cache hits)~%"
                      (getf cp :n-reused))

              (let ((n-eval (getf cp :n-eval))
                    (t-eval (getf cp :t-eval-ms)))
                (when (and (plusp n-eval) (plusp t-eval))
                  (format t "~%  Decode throughput: ~,1F tokens/s~%"
                          (/ n-eval (/ t-eval 1000.0)))))

              (format t "  Wall-clock elapsed: ~,1F ms~%" elapsed-ms)))

          ;; sampler-perf is separate — it takes the sampler chain pointer.
          (with-sampler-chain (chain)
            (let ((sp (sampler-perf chain)))
              (format t "~%  sampler-perf (fresh chain, no samples):~%")
              (format t "    :t-sample-ms ~,3F ms  :n-sample ~D~%"
                      (getf sp :t-sample-ms) (getf sp :n-sample))))

          (format t "~%✓ Structured perf data retrieved and throughput computed.~%")))))

  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  All phases complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
