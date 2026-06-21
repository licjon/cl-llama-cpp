;;; Model & context introspection example — prints a formatted report of all
;;; GGUF file-level info, model properties, metadata, and system capabilities.
;;;
;;; The GGUF sections run before loading the model (no backend needed).
;;; The model sections require a full load.
;;;
;;; Setup:
;;;   export LLAMA_MODEL=/path/to/model.gguf    ; or set *model-path* in the REPL
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/introspection.lisp")
;;;   (cl-llama-cpp/examples/introspection:run)

(defpackage #:cl-llama-cpp/examples/introspection
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/introspection)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun format-bytes (n)
  "Format byte count as a human-readable string."
  (cond ((>= n (* 1024 1024 1024))
         (format nil "~,2F GiB" (/ n (* 1024.0 1024.0 1024.0))))
        ((>= n (* 1024 1024))
         (format nil "~,2F MiB" (/ n (* 1024.0 1024.0))))
        ((>= n 1024)
         (format nil "~,2F KiB" (/ n 1024.0)))
        (t (format nil "~D bytes" n))))

(defun format-count (n)
  "Format large numbers with a suffix (B/M/K)."
  (cond ((>= n 1000000000) (format nil "~,2FB" (/ n 1.0e9)))
        ((>= n 1000000)    (format nil "~,2FM" (/ n 1.0e6)))
        ((>= n 1000)       (format nil "~,1FK" (/ n 1.0e3)))
        (t                 (format nil "~D" n))))

(defun format-gguf-val (val)
  "Format a gguf-metadata value for display."
  (cond ((and (consp val) (eq (car val) :array))
         (format nil "[array, ~D element~:P]" (cdr val)))
        ((eq val t)   "true")
        ((eq val nil) "false")
        ((floatp val) (format nil "~F" val))
        ((stringp val)
         (if (> (length val) 72)
             (concatenate 'string (subseq val 0 69) "...")
             val))
        (t (format nil "~A" val))))

;;; ── Main ─────────────────────────────────────────────────────────────

(defun run ()
  "Inspect a GGUF file and print a full introspection report."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (format t "~&File: ~A~%" *model-path*)

  ;; ── GGUF sections (no backend or model load needed) ───────────────
  (with-gguf (g *model-path* :no-alloc t)

    ;; ── File-level info ─────────────────────────────────────────────
    (banner "GGUF File Info")
    (format t "  GGUF version:      ~D~%"   (gguf-version g))
    (format t "  Alignment:         ~D bytes~%" (gguf-alignment g))
    (format t "  Data offset:       ~A~%"   (format-bytes (gguf-data-offset g)))
    (format t "  KV entries:        ~D~%"   (gguf-n-kv g))
    (format t "  Tensors:           ~D~%"   (gguf-n-tensors g))

    ;; ── Typed KV metadata ───────────────────────────────────────────
    (banner "GGUF Metadata (typed)")
    (let* ((meta (gguf-metadata g))
           (max-key-len (reduce #'max meta
                                :key (lambda (e) (length (car e)))
                                :initial-value 0)))
      (format t "  ~D entries:~2%" (length meta))
      (dolist (entry meta)
        (format t "  ~VA  ~A~%"
                (+ max-key-len 2) (car entry)
                (format-gguf-val (cdr entry)))))

    ;; ── Tensor listing ───────────────────────────────────────────────
    (banner "Tensors")
    (let* ((n     (gguf-n-tensors g))
           (show  (min n 16))
           (infos (loop for i from 0 below show collect (gguf-tensor-info g i)))
           (name-width (min 48 (reduce #'max infos
                                       :key (lambda (info) (length (getf info :name)))
                                       :initial-value 8))))
      (format t "  ~D tensor~:P total~2%" n)
      (format t "  ~VA  ~14A  ~A~%  ~A~%"
              name-width "Name" "Type" "Size"
              (make-string (+ name-width 30) :initial-element #\─))
      (dolist (info infos)
        (format t "  ~VA  ~14A  ~A~%"
                name-width
                (getf info :name)
                (getf info :type)
                (format-bytes (getf info :size))))
      (when (> n show)
        (format t "  ... (~D more)~%" (- n show)))))

  ;; ── Model sections (requires full load) ───────────────────────────
  (format t "~&~%Loading model weights...~%")
  (with-model (model *model-path*)
    (with-context (ctx model :n-ctx 2048)

      ;; ── Model description ────────────────────────────────────────
      (banner "Model Description")
      (format t "  ~A~%" (model-description model))

      ;; ── Model properties ─────────────────────────────────────────
      (banner "Model Properties")
      (let ((info (model-info model)))
        (format t "  Parameters:        ~A (~D)~%"
                (format-count (getf info :n-params)) (getf info :n-params))
        (format t "  File size:         ~A (~D bytes)~%"
                (format-bytes (getf info :size-bytes)) (getf info :size-bytes))
        (format t "  Layers:            ~D~%" (getf info :n-layers))
        (format t "  Attention heads:   ~D~%" (getf info :n-heads))
        (format t "  KV heads:          ~D~%" (getf info :n-heads-kv))
        (format t "  Embedding (in):    ~D~%" (getf info :n-embd-in))
        (format t "  Embedding (out):   ~D~%" (getf info :n-embd-out))
        (format t "  Training ctx:      ~D tokens~%" (getf info :n-ctx-train))
        (format t "  SWA size:          ~D~%" (getf info :n-swa))
        (format t "  RoPE type:         ~A~%" (getf info :rope-type))
        (format t "  RoPE freq scale:   ~F~%" (getf info :rope-freq-scale))
        (format t "  Classification outputs: ~D~%" (getf info :n-cls-out))
        (format t "~%  Architecture flags:~%")
        (format t "    Encoder:         ~:[no~;yes~]~%" (getf info :encoder-p))
        (format t "    Decoder:         ~:[no~;yes~]~%" (getf info :decoder-p))
        (format t "    Recurrent:       ~:[no~;yes~]~%" (getf info :recurrent-p))
        (format t "    Hybrid:          ~:[no~;yes~]~%" (getf info :hybrid-p))
        (format t "    Diffusion:       ~:[no~;yes~]~%" (getf info :diffusion-p)))

      ;; ── Context configuration ────────────────────────────────────
      (banner "Context Configuration")
      (let ((info (context-info ctx)))
        (format t "  Context size:      ~D tokens~%" (getf info :n-ctx))
        (format t "  Batch size:        ~D~%" (getf info :n-batch))
        (format t "  Micro-batch size:  ~D~%" (getf info :n-ubatch))
        (format t "  Max sequences:     ~D~%" (getf info :n-seq-max))
        (format t "  Threads:           ~D~%" (getf info :n-threads))
        (format t "  Threads (batch):   ~D~%" (getf info :n-threads-batch))
        (format t "  Pooling type:      ~A~%" (getf info :pooling-type)))

      ;; ── Model metadata (string view via llama.cpp meta API) ──────
      (banner "Model Metadata (string view)")
      (let ((metadata (model-metadata model)))
        (format t "  ~D entries:~2%" (length metadata))
        (let ((max-key-len (reduce #'max metadata
                                   :key (lambda (e) (length (car e)))
                                   :initial-value 0)))
          (dolist (entry metadata)
            (let ((val (cdr entry)))
              (format t "  ~VA  ~A~%"
                      (+ max-key-len 2) (car entry)
                      (if (> (length val) 72)
                          (concatenate 'string (subseq val 0 69) "...")
                          val))))))

      ;; ── System info ──────────────────────────────────────────────
      (banner "System / Build Info")
      (format t "  ~A~%" (system-info))))

  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  Report complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
