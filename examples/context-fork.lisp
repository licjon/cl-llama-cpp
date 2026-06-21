;;; Context forking example — process a prompt once, then branch into
;;; multiple alternative continuations from the same saved state.
;;;
;;; This demonstrates the key benefit of state save/load: you pay the
;;; prompt-processing cost once and explore many paths.  Use cases include
;;; brainstorming alternatives, A/B testing sampler settings, speculative
;;; decoding, and "choose your own adventure" style branching.
;;;
;;; Setup:
;;;   export LLAMA_MODEL=/path/to/model.gguf    ; or set *model-path* in the REPL
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/context-fork.lisp")
;;;   (cl-llama-cpp/examples/context-fork:run)

(defpackage #:cl-llama-cpp/examples/context-fork
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/context-fork)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun section (title)
  (format t "~&~%── ~A ──~2%" title))

;;; ── Main ─────────────────────────────────────────────────────────────

(defun run ()
  "Process a prompt once, snapshot the state, then generate multiple
alternative continuations from the same point."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (format t "~&Loading model: ~A~%" *model-path*)
  (with-model (model *model-path* :n-gpu-layers 99)
    (with-context (ctx model :n-ctx 2048)
      (let ((prompt "Once upon a time, in a kingdom where code was law,"))

        ;; ═════════════════════════════════════════════════════════════
        (banner "PHASE 1: Process Prompt & Snapshot")
        ;; ═════════════════════════════════════════════════════════════

        (format t "Prompt: ~S~2%" prompt)

        ;; Tokenize and decode the prompt into the KV cache
        (let ((tokens (tokenize model prompt :parse-special t)))
          (format t "Tokenized to ~D tokens.~%" (length tokens))
          (with-batch (batch (length tokens))
            (batch-add-sequence batch tokens 0 :logits :last)
            (batch-decode ctx batch))
          (format t "Decoded into KV cache.~%")

          ;; Snapshot the context state to memory
          (let ((snapshot (save-state ctx)))
            (format t "~%State snapshot: ~:D bytes captured.~%" (length snapshot))
            (assert (> (length snapshot) 0) ()
                    "Snapshot should be non-empty after decoding")

            ;; ═════════════════════════════════════════════════════════
            (banner "PHASE 2: Fork — Multiple Continuations")
            ;; ═════════════════════════════════════════════════════════

            ;; Each branch restores the same snapshot, then generates
            ;; with different sampler settings.  The prompt is NOT
            ;; re-decoded — that work is reused from the snapshot.

            (let ((branches
                    '((:label "Conservative (temp=0.1)"
                       :temp 0.1 :top-k nil :top-p nil)
                      (:label "Creative (temp=1.2, top-p=0.95)"
                       :temp 1.2 :top-k nil :top-p 0.95)
                      (:label "Focused (temp=0.5, top-k=10)"
                       :temp 0.5 :top-k 10 :top-p nil)))
                  (results nil))

              (dolist (branch branches)
                (let ((label (getf branch :label))
                      (temp  (getf branch :temp))
                      (top-k (getf branch :top-k))
                      (top-p (getf branch :top-p)))

                  (section label)

                  ;; Restore the snapshot — rewinds to right after prompt decode
                  (let ((bytes-read (load-state ctx snapshot)))
                    (format t "Restored ~:D bytes of state.~%" bytes-read)
                    (assert (> bytes-read 0) ()
                            "load-state should consume bytes"))

                  ;; Generate from the restored state.  We pass :prompt-tokens
                  ;; so generate knows the prompt length but skips re-decoding.
                  (let ((text (generate ctx nil
                                :prompt-tokens tokens
                                :max-tokens 64
                                :temp temp
                                :top-k top-k
                                :top-p top-p
                                :seed 42)))
                    (format t "~%  \"~A~A\"~%" prompt text)
                    (push (cons label text) results))))

              (format t "~%All ~D branches started from the same snapshot.~%"
                      (length branches))
              (format t "The prompt was decoded only once.~%")

              ;; ═════════════════════════════════════════════════════════
              (banner "PHASE 3: File-Based Persistence")
              ;; ═════════════════════════════════════════════════════════

              ;; Save state to disk, clear everything, reload, and show
              ;; that the restored state generates identically.

              ;; First: generate from the in-memory snapshot (the "before")
              (load-state ctx snapshot)
              (let* ((gen-params '(:max-tokens 32 :temp 0.3 :seed 99))
                     (before-text (apply #'generate ctx nil
                                         :prompt-tokens tokens gen-params)))
                (format t "Generated before save:~%")
                (format t "  \"~A~A\"~2%" prompt before-text)

                ;; Save to disk
                (load-state ctx snapshot)
                (let ((session-path (namestring
                                     (merge-pathnames "cl-llama-session.bin"
                                                      (uiop:temporary-directory)))))
                  (format t "Saving session to: ~A~%" session-path)
                  (save-session ctx session-path tokens)
                  (assert (probe-file session-path) ()
                          "Session file should exist on disk")
                  (format t "Session file written (~:D bytes on disk).~%"
                          (with-open-file (s session-path) (file-length s)))

                  ;; Nuke everything, then reload
                  (clear-kv-cache ctx)
                  (format t "~%KV cache cleared.~%")

                  (let ((loaded-tokens (load-session ctx session-path)))
                    (format t "Session loaded — ~D tokens recovered.~%"
                            (length loaded-tokens))
                    (assert (= (length tokens) (length loaded-tokens)) ()
                            "Token count mismatch: ~D vs ~D"
                            (length tokens) (length loaded-tokens))
                    (assert (equalp tokens loaded-tokens) ()
                            "Token values should match after roundtrip")

                    ;; Generate from restored state with identical params
                    (let ((after-text (apply #'generate ctx nil
                                             :prompt-tokens loaded-tokens
                                             gen-params)))
                      (format t "~%Generated after load:~%")
                      (format t "  \"~A~A\"~2%" prompt after-text)

                      (if (string= before-text after-text)
                          (format t "Outputs match — file roundtrip preserved state exactly.~%")
                          (format t "Note: outputs differ (expected with some backends).~%"))))

                  ;; Cleanup
                  (delete-file session-path)
                  (format t "Temp file cleaned up.~%"))))))))

    (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
    (format t "  All assertions passed.~%")
    (format t "~A~%" (make-string 64 :initial-element #\═))
    (values)))
