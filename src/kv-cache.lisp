(in-package #:cl-llama-cpp)

;;; KV cache / memory management wrappers

(llama-defun clear-kv-cache (ctx)
  "Clear all KV cache state for CTX."
  (%llama:memory-clear (%llama:get-memory (llama-context-pointer ctx)) 1)
  nil)

(llama-defun kv-cache-seq-rm (ctx seq-id p0 p1)
  "Remove cached tokens in positions [P0, P1) for SEQ-ID.
P0=-1 means from the start, P1=-1 means to the end.
Returns T if cells were removed, NIL if no matching data."
  (not (zerop (%llama:memory-seq-rm
               (%llama:get-memory (llama-context-pointer ctx)) seq-id p0 p1))))

(llama-defun kv-cache-seq-cp (ctx src-seq dst-seq p0 p1)
  "Copy cached data from SRC-SEQ to DST-SEQ for positions [P0, P1)."
  (%llama:memory-seq-cp (%llama:get-memory (llama-context-pointer ctx)) src-seq dst-seq p0 p1)
  nil)

(llama-defun kv-cache-seq-keep (ctx seq-id)
  "Keep only SEQ-ID's cached data, removing all other sequences."
  (%llama:memory-seq-keep (%llama:get-memory (llama-context-pointer ctx)) seq-id)
  nil)

(llama-defun kv-cache-seq-add (ctx seq-id p0 p1 delta)
  "Shift positions in [P0, P1) for SEQ-ID by DELTA."
  (%llama:memory-seq-add (%llama:get-memory (llama-context-pointer ctx)) seq-id p0 p1 delta)
  nil)

(llama-defun kv-cache-seq-div (ctx seq-id p0 p1 d)
  "Divide positions in [P0, P1) for SEQ-ID by D. D must be non-zero.
Signals INPUT-VALIDATION-ERROR if D is zero."
  (check-type d integer)
  (when (zerop d)
    (error 'input-validation-error
           :function-name 'kv-cache-seq-div :argument :d :value d
           :reason "divisor must be non-zero"))
  (%llama:memory-seq-div (%llama:get-memory (llama-context-pointer ctx)) seq-id p0 p1 d)
  nil)

(llama-defun kv-cache-pos (ctx seq-id)
  "Return the minimum and maximum cached positions for SEQ-ID as (VALUES MIN MAX)."
  (let ((mem (%llama:get-memory (llama-context-pointer ctx))))
    (values (%llama:memory-seq-pos-min mem seq-id)
            (%llama:memory-seq-pos-max mem seq-id))))

(llama-defun kv-cache-can-shift-p (ctx)
  "Return T if CTX's memory supports position shifting, NIL otherwise."
  (not (zerop (%llama:memory-can-shift (%llama:get-memory (llama-context-pointer ctx))))))
