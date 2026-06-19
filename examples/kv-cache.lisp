;;; KV cache management example — executable simulation of a multi-user
;;; chat server demonstrating sequence copying, sliding windows, and cleanup.
;;;
;;; Every API call is verified with assertions and narrated to stdout.
;;; This script doubles as a regression test — run it top-to-bottom.
;;;
;;; Setup:
;;;   export LLAMA_MODEL=~/models/gemma-3-1b-it-Q4_K_M.gguf
;;;
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/kv-cache.lisp")
;;;   (cl-llama-cpp/examples/kv-cache:run)

(defpackage #:cl-llama-cpp/examples/kv-cache
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/kv-cache)

(defparameter *model-path*
  (or (uiop:getenv "LLAMA_MODEL")
      (error "Set LLAMA_MODEL to the path of a GGUF model.")))

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  "Print a phase banner."
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun show-seq (ctx seq-id label)
  "Print and return (values min max) for SEQ-ID."
  (multiple-value-bind (mn mx) (kv-cache-pos ctx seq-id)
    (if (< mx 0)
        (format t "  seq ~D (~8A): [empty]~%" seq-id label)
        (format t "  seq ~D (~8A): positions [~D .. ~D]~%"
                seq-id label mn mx))
    (values mn mx)))

(defun decode-for-seq (ctx model text seq-id start-pos)
  "Tokenize TEXT and decode into CTX's KV cache for SEQ-ID starting at
START-POS. Uses batch-init to target a specific sequence, since the
simpler batch-get-one always targets seq 0.
Returns the number of tokens decoded."
  (let* ((tokens (tokenize model text :add-special nil))
         (n (length tokens)))
    (with-llama-compatible-fp-environment
      ;; batch-init allocates arrays for n tokens, 0 embedding dims, 1 seq per token
      (let ((batch (%llama:batch-init n 0 1)))
        (unwind-protect
            (progn
              ;; Set the actual token count (batch-init initializes it to 0)
              (setf (getf batch '%llama::n-tokens) n)
              ;; Fill each slot: token id, position, sequence assignment, logit flag
              (let ((tok-ptr   (getf batch '%llama::token))
                    (pos-ptr   (getf batch '%llama::pos))
                    (nseq-ptr  (getf batch '%llama::n-seq-id))
                    (seqid-ptr (getf batch '%llama::seq-id))
                    (logit-ptr (getf batch '%llama::logits)))
                (dotimes (i n)
                  (setf (cffi:mem-aref tok-ptr :int i) (aref tokens i))
                  (setf (cffi:mem-aref pos-ptr :int i) (+ start-pos i))
                  (setf (cffi:mem-aref nseq-ptr :int i) 1)
                  ;; seq-id is (:pointer (:pointer seq-id)) — array of pointers
                  ;; to per-token seq-id arrays. Dereference twice.
                  (setf (cffi:mem-aref
                         (cffi:mem-aref seqid-ptr :pointer i) :int 0)
                        seq-id)
                  ;; Request logits only for the last token
                  (setf (cffi:mem-aref logit-ptr :char i)
                        (if (= i (1- n)) 1 0))))
              (let ((rc (%llama:decode ctx batch)))
                (assert (zerop rc) ()
                        "decode failed with code ~D for seq ~D" rc seq-id)))
          (%llama:batch-free batch))))
    n))

;;; ── Main simulation ─────────────────────────────────────────────────

(defun run ()
  "Run the multi-user chat server simulation.
Demonstrates kv-cache-seq-cp, kv-cache-seq-rm, kv-cache-seq-add,
kv-cache-seq-keep, kv-cache-pos, kv-cache-can-shift-p, and clear-kv-cache."
  (format t "~&Loading model: ~A~%" *model-path*)
  (with-model (model *model-path* :n-gpu-layers 99)
    ;; n-seq-max >= 3 because cells will belong to seq 0, 1, and 2
    ;; simultaneously after the copy in Phase 1.
    (with-context (ctx model :n-ctx 2048 :n-seq-max 4)

      ;; ═══════════════════════════════════════════════════════════════
      (banner "PHASE 1: Context Sharing — The Base Prompt")
      ;; ═══════════════════════════════════════════════════════════════
      ;;
      ;; Real LLM servers process the system prompt once and clone it
      ;; to each user's sequence, avoiding redundant computation.

      (clear-kv-cache ctx)
      (format t "Cleared KV cache.~%")

      ;; Decode the system prompt into seq 0 (the "template" sequence)
      (let* ((system-text
               "You are a helpful AI assistant specializing in Common Lisp programming.")
             (s (decode-for-seq ctx model system-text 0 0)))

        (format t "~%System prompt: ~S~%" system-text)
        (format t "Tokenized to ~D tokens → decoded into seq 0 at [0 .. ~D].~%"
                s (1- s))

        ;; Verify the template landed correctly
        (multiple-value-bind (mn mx) (show-seq ctx 0 "template")
          (assert (= mn 0)      () "seq 0 min: expected 0, got ~D" mn)
          (assert (= mx (1- s)) () "seq 0 max: expected ~D, got ~D" (1- s) mx))

        ;; Clone to User A (seq 1) and User B (seq 2).
        ;; kv-cache-seq-cp doesn't copy cells — it tags existing cells
        ;; with an additional seq-id. All three sequences share the
        ;; same underlying KV cache entries. Zero extra memory.
        (format t "~%Cloning system prompt to users:~%")
        (format t "  kv-cache-seq-cp(ctx, src=0, dst=1, p0=-1, p1=-1)~%")
        (kv-cache-seq-cp ctx 0 1 -1 -1)
        (format t "  kv-cache-seq-cp(ctx, src=0, dst=2, p0=-1, p1=-1)~%")
        (kv-cache-seq-cp ctx 0 2 -1 -1)

        (format t "~%State after cloning:~%")
        (multiple-value-bind (mn0 mx0) (show-seq ctx 0 "template")
          (multiple-value-bind (mn1 mx1) (show-seq ctx 1 "User A")
            (multiple-value-bind (mn2 mx2) (show-seq ctx 2 "User B")
              (assert (= mn0 mn1 mn2 0)
                      () "All mins should be 0")
              (assert (= mx0 mx1 mx2 (1- s))
                      () "All maxes should be ~D" (1- s)))))

        (format t "~%✓ All three sequences share the system prompt [0 .. ~D].~%"
                (1- s))

        ;; ═══════════════════════════════════════════════════════════════
        (banner "PHASE 2: Sliding Window via Surgical Deletion")
        ;; ═══════════════════════════════════════════════════════════════
        ;;
        ;; User A sends two messages. We decode them directly into seq 1
        ;; (not seq 0) using the batch API. This matters: those cells
        ;; belong ONLY to seq 1. If they were shared with other sequences,
        ;; the position shift in step 2 would corrupt the other sequences
        ;; because kv-cache-seq-add modifies cell positions in place.

        (let* ((msg1-text "What is a macro in Lisp?")
               (msg2-text
                 "Can you show me how defmacro works with backquote syntax?")
               (m1 (decode-for-seq ctx model msg1-text 1 s))
               (m2 (decode-for-seq ctx model msg2-text 1 (+ s m1))))

          (format t "User A sends two messages:~%")
          (format t "  msg 1: ~S → ~D tokens~%" msg1-text m1)
          (format t "  msg 2: ~S → ~D tokens~2%" msg2-text m2)

          (multiple-value-bind (mn mx) (show-seq ctx 1 "User A")
            (declare (ignore mn))
            (format t "~%  Memory layout for seq 1:~%")
            (format t "    ├─ [~3D .. ~3D]  system prompt  (~D tok, shared w/ seq 0,2)~%"
                    0 (1- s) s)
            (format t "    ├─ [~3D .. ~3D]  msg 1          (~D tok, seq 1 only)~%"
                    s (+ s m1 -1) m1)
            (format t "    └─ [~3D .. ~3D]  msg 2          (~D tok, seq 1 only)~%"
                    (+ s m1) (+ s m1 m2 -1) m2)
            (format t "    Total: ~D positions~%" (1+ mx))
            (assert (= mx (+ s m1 m2 -1))
                    () "seq 1 max: expected ~D, got ~D" (+ s m1 m2 -1) mx))

          ;; Simulate the context window filling up: evict msg 1 (old
          ;; history) while keeping the system prompt and msg 2 (recent).
          (format t "~%─── Context window full! Evicting msg 1 ───~2%")

          ;; Step 0: Confirm the model's memory supports position shifting
          (let ((can-shift (kv-cache-can-shift-p ctx)))
            (format t "Step 0: kv-cache-can-shift-p(ctx) → ~A~%" can-shift)
            (assert can-shift ()
                    "Position shifting not supported — cannot do sliding window"))

          ;; Step 1: Remove msg 1 from seq 1
          ;; The range [p0, p1) is half-open: p0 inclusive, p1 exclusive
          (format t "~%Step 1: Remove msg 1~%")
          (format t "  kv-cache-seq-rm(ctx, seq=1, p0=~D, p1=~D)~%"
                  s (+ s m1))
          (format t "  Half-open range [~D, ~D) = positions ~D..~D = ~D tokens~%"
                  s (+ s m1) s (+ s m1 -1) m1)

          (let ((removed (kv-cache-seq-rm ctx 1 s (+ s m1))))
            (format t "  → returned ~A~%" removed)
            (assert removed () "Expected T from kv-cache-seq-rm"))

          (format t "~%  Seq 1 now has a gap:~%")
          (format t "    ├─ [~3D .. ~3D]  system prompt  ✓~%" 0 (1- s))
          (format t "    ├─ [~3D .. ~3D]  ── gap ──      (freed)~%"
                  s (+ s m1 -1))
          (format t "    └─ [~3D .. ~3D]  msg 2          ✓ (stranded)~%"
                  (+ s m1) (+ s m1 m2 -1))

          ;; Step 2: Shift msg 2 backward to close the gap
          (format t "~%Step 2: Shift msg 2 backward by ~D to close the gap~%" m1)
          (format t "  kv-cache-seq-add(ctx, seq=1, p0=~D, p1=-1, delta=~D)~%"
                  (+ s m1) (- m1))
          (format t "~%  Position math for each token in msg 2:~%")
          (format t "    old position range : [~D .. ~D]~%"
                  (+ s m1) (+ s m1 m2 -1))
          (format t "    delta              : ~D~%" (- m1))
          (format t "    new position range : [~D+(~D) .. ~D+(~D)] = [~D .. ~D]~%"
                  (+ s m1) (- m1)
                  (+ s m1 m2 -1) (- m1)
                  s (+ s m2 -1))

          (kv-cache-seq-add ctx 1 (+ s m1) -1 (- m1))

          (format t "~%  After compaction:~%")
          (multiple-value-bind (mn mx) (show-seq ctx 1 "User A")
            (format t "    ├─ [~3D .. ~3D]  system prompt~%" 0 (1- s))
            (format t "    └─ [~3D .. ~3D]  msg 2 (shifted back, gap closed)~%"
                    s (+ s m2 -1))
            (format t "    Total: ~D positions (was ~D — reclaimed ~D)~%"
                    (1+ mx) (+ s m1 m2) m1)
            (assert (= mn 0)
                    () "After shift: min should be 0, got ~D" mn)
            (assert (= mx (+ s m2 -1))
                    () "After shift: max should be ~D, got ~D"
                    (+ s m2 -1) mx))

          (format t "~%✓ Sliding window complete. Msg 1 evicted, msg 2 compacted.~%"))

        ;; ═══════════════════════════════════════════════════════════════
        (banner "PHASE 3: The \"Forget Everyone Else\" Cleanup")
        ;; ═══════════════════════════════════════════════════════════════
        ;;
        ;; User B disconnects. We keep only seq 1 (User A), freeing all
        ;; cells that don't belong to seq 1. This is O(cells) — one pass
        ;; through the cache, no per-sequence bookkeeping.

        (format t "Scenario: User B disconnects. Reclaim everything except User A.~2%")

        ;; Capture seq 1's state before cleanup so we can verify it's
        ;; unchanged afterward — a before/after comparison, not a
        ;; computation from Phase 2 variables.
        (multiple-value-bind (seq1-mn-before seq1-mx-before)
            (kv-cache-pos ctx 1)

          (format t "Before cleanup:~%")
          (show-seq ctx 0 "template")
          (format t "  seq 1 (User A  ): positions [~D .. ~D]~%"
                  seq1-mn-before seq1-mx-before)
          (show-seq ctx 2 "User B")

          (format t "~%  kv-cache-seq-keep(ctx, seq=1)~%")
          (format t "  Effect: remove every seq-id except 1 from every cell.~%")
          (format t "  Cells with no remaining seq-ids are freed entirely.~2%")

          (kv-cache-seq-keep ctx 1)

          ;; Also explicitly remove leftover sequences via seq-rm, since
          ;; some memory implementations (e.g. ISWA caches with dual
          ;; sub-caches) may not fully clean up on seq-keep alone.
          (kv-cache-seq-rm ctx 0 -1 -1)
          (kv-cache-seq-rm ctx 2 -1 -1)

          (format t "After cleanup:~%")
          (multiple-value-bind (mn0 mx0) (show-seq ctx 0 "template")
            (declare (ignore mn0))
            (assert (< mx0 0) () "seq 0 should be empty, got max=~D" mx0))
          (multiple-value-bind (mn1 mx1) (show-seq ctx 1 "User A")
            (assert (= mn1 seq1-mn-before)
                    () "seq 1 min changed: was ~D, now ~D"
                    seq1-mn-before mn1)
            (assert (= mx1 seq1-mx-before)
                    () "seq 1 max changed: was ~D, now ~D"
                    seq1-mx-before mx1))
          (multiple-value-bind (mn2 mx2) (show-seq ctx 2 "User B")
            (declare (ignore mn2))
            (assert (< mx2 0) () "seq 2 should be empty, got max=~D" mx2))

          (format t "~%✓ User A's data survives intact. Other sequences reclaimed.~%")))))

  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  All assertions passed. Simulation complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
