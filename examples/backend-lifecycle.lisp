;;; Backend lifecycle and runtime configuration example.
;;;
;;; with-backend       — proper init/free lifecycle (replaces bare ensure-backend)
;;; set-n-threads      — adjust decode and batch thread counts at runtime
;;; set-warmup         — disable the warmup pass on a cold context
;;; set-causal-attn    — toggle causal vs bidirectional attention
;;; synchronize        — drain pending async compute before inspecting state
;;; set-abort-callback — interrupt long-running inference via a deadline
;;;
;;; Setup:
;;;   export LLAMA_MODEL=~/models/gemma-3-1b-it-Q4_K_M.gguf
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/backend-lifecycle.lisp")
;;;   (cl-llama-cpp/examples/backend-lifecycle:run)

(defpackage #:cl-llama-cpp/examples/backend-lifecycle
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/backend-lifecycle)

(defparameter *model-path*
  (or (uiop:getenv "LLAMA_MODEL")
      (error "Set LLAMA_MODEL to the path of a GGUF model.")))

;;; ── Abort callback ───────────────────────────────────────────────────
;;;
;;; defcallback must be at the top level so cffi:callback returns a
;;; stable pointer. The void* data argument IS the flag — the callback
;;; reads a single :uint8 byte from it. Foreign memory is visible from
;;; any C or Lisp thread, unlike CL dynamic bindings which are
;;; thread-local and not inherited by ggml worker threads.

(cffi:defcallback deadline-callback :bool ((flag :pointer))
  (not (zerop (cffi:mem-ref flag :uint8))))

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun show-thread-config (ctx label)
  (let ((info (context-info ctx)))
    (format t "  ~A: n-threads=~D  n-threads-batch=~D~%"
            label
            (getf info :n-threads)
            (getf info :n-threads-batch))))

;;; ── Main ─────────────────────────────────────────────────────────────

(defun run ()
  "Run all backend-lifecycle demonstrations."
  (format t "~&Loading model: ~A~%" *model-path*)

  ;; with-backend is the proper replacement for bare ensure-backend calls.
  ;; It guarantees backend-free runs on exit even on non-local transfer of
  ;; control. Nesting is safe: only the outermost call shuts down the backend.
  (with-backend ()

    ;; ══════════════════════════════════════════════════════════════════
    (banner "PHASE 1: Thread count tuning")
    ;; ══════════════════════════════════════════════════════════════════
    ;;
    ;; Thread counts can be changed at any time without recreating the
    ;; context. set-n-threads takes both counts together because the C API
    ;; requires them in one call — they are not independently settable.

    (with-model (model *model-path* :n-gpu-layers 0)
      (with-context (ctx model :n-ctx 512)

        (format t "Initial thread configuration from context-info:~%")
        (show-thread-config ctx "default")

        (let ((info (context-info ctx)))
          (let ((orig-n  (getf info :n-threads))
                (orig-nb (getf info :n-threads-batch)))

            ;; Increase decode threads for a latency-sensitive workload.
            (let ((new-n  (max 1 (* 2 orig-n)))
                  (new-nb (max 1 (floor orig-nb 2))))
              (format t "~%Calling (set-n-threads ctx ~D ~D)...~%" new-n new-nb)
              (set-n-threads ctx new-n new-nb)
              (show-thread-config ctx "after set-n-threads")

              (let ((info2 (context-info ctx)))
                (assert (= (getf info2 :n-threads) new-n)
                        () "n-threads not updated: expected ~D got ~D"
                        new-n (getf info2 :n-threads))
                (assert (= (getf info2 :n-threads-batch) new-nb)
                        () "n-threads-batch not updated"))

              ;; Restore original counts.
              (set-n-threads ctx orig-n orig-nb)
              (show-thread-config ctx "restored")

              (format t "~%✓ Thread counts changed and restored without context rebuild.~%"))))

        ;; ══════════════════════════════════════════════════════════════
        (banner "PHASE 2: Context mode toggles")
        ;; ══════════════════════════════════════════════════════════════
        ;;
        ;; set-warmup and set-causal-attn each flip a single boolean flag
        ;; on the context. They take effect on the next decode.
        ;;
        ;; set-embeddings is omitted: it must match the mode the context
        ;; was created with (see the set-embeddings docstring).

        (format t "Disabling warmup pass (useful after the first decode):~%")
        (set-warmup ctx nil)
        (format t "  (set-warmup ctx nil) → ok~%")

        (format t "~%Toggling causal attention off and back on:~%")
        (set-causal-attn ctx nil)
        (format t "  (set-causal-attn ctx nil) → ok~%")
        (set-causal-attn ctx t)
        (format t "  (set-causal-attn ctx t)   → ok~%")

        (format t "~%✓ Mode toggles applied without recreating the context.~%")

        ;; ══════════════════════════════════════════════════════════════
        (banner "PHASE 3: Synchronize")
        ;; ══════════════════════════════════════════════════════════════
        ;;
        ;; synchronize is a barrier: it blocks until the GPU (or async
        ;; CPU backend) finishes pending compute. Call it before reading
        ;; logits or embeddings if you have issued non-blocking work.

        (format t "Calling (synchronize ctx) as a compute barrier:~%")
        (synchronize ctx)
        (format t "  → done (all pending async operations complete)~%")
        (format t "~%✓ synchronize returned cleanly.~%")

        ;; ══════════════════════════════════════════════════════════════
        (banner "PHASE 4: Time-limited generation via abort callback")
        ;; ══════════════════════════════════════════════════════════════
        ;;
        ;; The callback reads a :uint8 from the flag pointer passed as
        ;; data — foreign memory is visible from ggml worker threads,
        ;; unlike CL dynamic bindings which are thread-local. A Lisp
        ;; timer sets the flag after 2 seconds.

        (let ((abort-flag (cffi:foreign-alloc :uint8 :initial-element 0)))
          (unwind-protect
              (progn
                ;; Pass the flag pointer as data; the callback reads it directly.
                (set-abort-callback ctx (cffi:callback deadline-callback) abort-flag)
                (format t "  Callback registered. Arming 2-second deadline...~%")

                (let* ((timer (sb-ext:make-timer
                                (lambda ()
                                  (setf (cffi:mem-ref abort-flag :uint8) 1))))
                       (started (get-internal-real-time))
                       (stop-reason nil))
                  (sb-ext:schedule-timer timer 2.0)
                  (unwind-protect
                      (handler-case
                          (multiple-value-bind (text reason)
                              (generate ctx "Count from one:" :max-tokens 512 :temp 0.1)
                            (setf stop-reason reason)
                            (format t "  Generated (stop=~A): ~S~%"
                                    reason
                                    (if (> (length text) 60)
                                        (concatenate 'string (subseq text 0 57) "...")
                                        text)))
                        (error (e)
                          (setf stop-reason :aborted)
                          (format t "  Generation aborted by callback: ~A~%" e)))
                    (sb-ext:unschedule-timer timer))

                  (let ((elapsed (/ (- (get-internal-real-time) started)
                                    internal-time-units-per-second)))
                    (format t "  Elapsed: ~,2F s  stop-reason: ~A~%"
                            elapsed (or stop-reason :aborted)))

                  (set-abort-callback ctx nil)
                  (format t "  Callback cleared.~%")
                  (format t "~%✓ Inference interrupted by deadline callback.~%")))
            (cffi:foreign-free abort-flag)))))

    (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
    (format t "  All phases complete. Backend freed by with-backend.~%")
    (format t "~A~%" (make-string 64 :initial-element #\═)))

  (values))
