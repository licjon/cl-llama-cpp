;;; Hot-path benchmark — tokenize, detokenize, generate, embed.
;;;
;;; Run with models set to compare before/after type declarations:
;;;
;;;   LLAMA_TEST_MODEL=~/models/gemma-3-1b-it-Q4_K_M.gguf \
;;;   LLAMA_TEST_EMBED_MODEL=~/models/nomic-embed-text-v2-moe-q8_0.gguf \
;;;   ros -l examples/benchmark-hotpaths.lisp -e '(cl-llama-cpp/examples/benchmark-hotpaths:run)'
;;;
;;; Prints ms/op and tokens/sec for each section.

(defpackage #:cl-llama-cpp/examples/benchmark-hotpaths
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/benchmark-hotpaths)

(defvar *model-path*       (uiop:getenv "LLAMA_TEST_MODEL"))
(defvar *embed-model-path* (uiop:getenv "LLAMA_TEST_EMBED_MODEL"))

(defparameter *bench-text*
  "The quick brown fox jumps over the lazy dog. \
Pack my box with five dozen liquor jugs. \
How vexingly quick daft zebras jump! \
The five boxing wizards jump quickly.")

(defparameter *n-tokenize-reps* 1000)
(defparameter *n-embed-reps*    50)
(defparameter *generate-tokens* 64)

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun ms-since (start)
  (* 1000.0d0 (/ (- (get-internal-real-time) start)
                 (float internal-time-units-per-second 1.0d0))))

(defun banner (title)
  (format t "~&~%~A~%" (make-string 60 :initial-element #\─))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 60 :initial-element #\─)))

;;; ── Tokenize / detokenize bench ──────────────────────────────────────

(defun bench-tokenization (model)
  (banner "Tokenize / Detokenize")
  ;; Warm up
  (let ((tokens (tokenize model *bench-text*)))
    (detokenize model tokens))
  (let* ((start (get-internal-real-time))
         (n-total-tokens 0))
    (dotimes (i *n-tokenize-reps*)
      (let ((tokens (tokenize model *bench-text*)))
        (incf n-total-tokens (length tokens))))
    (let* ((elapsed (ms-since start))
           (ms-per-op (/ elapsed *n-tokenize-reps*))
           (tok-per-sec (* 1000.0d0 (/ n-total-tokens elapsed))))
      (format t "tokenize: ~D reps, ~,2F ms/op, ~,1F tok/s~%"
              *n-tokenize-reps* ms-per-op tok-per-sec)))
  (let* ((tokens (tokenize model *bench-text*))
         (start (get-internal-real-time)))
    (dotimes (i *n-tokenize-reps*)
      (detokenize model tokens))
    (let* ((elapsed (ms-since start))
           (ms-per-op (/ elapsed *n-tokenize-reps*)))
      (format t "detokenize: ~D reps, ~,2F ms/op~%"
              *n-tokenize-reps* ms-per-op))))

;;; ── Generate bench ───────────────────────────────────────────────────

(defun bench-generate (model ctx)
  (banner "Generate")
  ;; Warm up
  (generate ctx *bench-text* :max-tokens 4 :temp 0.0)
  (let* ((start (get-internal-real-time))
         (result (generate ctx *bench-text*
                            :max-tokens *generate-tokens*
                            :temp 0.0))
         (elapsed (ms-since start))
         (n-tok (length (tokenize model result))))
    (format t "generate (~D max-tokens): ~,1F ms total, ~,1F gen-tok/s~%"
            *generate-tokens* elapsed
            (* 1000.0d0 (/ n-tok elapsed)))))

;;; ── Embed bench ──────────────────────────────────────────────────────

(defun bench-embed (ctx)
  (banner "Embed")
  ;; Warm up
  (embed ctx *bench-text*)
  (let* ((start (get-internal-real-time)))
    (dotimes (i *n-embed-reps*)
      (embed ctx *bench-text*))
    (let* ((elapsed (ms-since start))
           (ms-per-op (/ elapsed *n-embed-reps*)))
      (format t "embed: ~D reps, ~,2F ms/op~%"
              *n-embed-reps* ms-per-op))))

;;; ── Entry point ──────────────────────────────────────────────────────

(defun run ()
  (format t "~&cl-llama-cpp hot-path benchmark~%")
  (format t "text length: ~D chars~%" (length *bench-text*))

  (if *model-path*
      (with-model (model *model-path* :n-gpu-layers 99)
        (bench-tokenization model)
        (with-context (ctx model)
          (bench-generate model ctx)))
      (format t "~&[skip] LLAMA_TEST_MODEL not set — tokenize/generate skipped~%"))

  (if *embed-model-path*
      (with-model (embed-model *embed-model-path* :n-gpu-layers 99)
        (with-context (embed-ctx embed-model :embeddings t)
          (bench-embed embed-ctx)))
      (format t "~&[skip] LLAMA_TEST_EMBED_MODEL not set — embed skipped~%"))

  (format t "~&done.~%"))
