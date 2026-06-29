(in-package #:cl-llama-cpp)

;;; Incremental chat session — reuses the KV cache across turns so each turn
;;; decodes only the *new* tokens (the delta), not the entire conversation.
;;;
;;; The core invariant: CHAT-SESSION-DECODED holds the exact token sequence
;;; that is currently resident in sequence 0 of the context's KV cache.  On
;;; each call to CHAT-SESSION-SEND the full prompt is re-rendered via
;;; TOKENIZE-CHAT, the common prefix with DECODED is found, any stale suffix
;;; is evicted from the cache, and only the delta (new tokens) plus the
;;; assistant-prefix are decoded.  This keeps per-turn prefill cost constant
;;; regardless of conversation length.

(defstruct (chat-session (:constructor %make-chat-session) (:copier nil))
  "Stateful multi-turn chat that reuses the KV cache across turns.
CTX and MODEL are borrowed (not owned) — the caller is responsible for their
lifetime.  DECODED is the exact token vector currently resident in sequence 0
of the KV cache, maintained as an invariant by CHAT-SESSION-SEND.

MESSAGES is the source of truth for the conversation.  Callers may append to,
truncate, or otherwise edit MESSAGES between turns; the next CHAT-SESSION-SEND
reconciles the KV cache by re-rendering the full prompt, diffing against the
decoded prefix, evicting any divergent suffix, and decoding only the delta.
Editing earlier turns invalidates the cache from the first changed token onward
\(correct, but requires re-decoding from that point)."
  (context  nil :type (or llama-context null) :read-only t)
  (model    nil :type (or llama-model null)   :read-only t)
  (template nil :type (or string null))
  (messages nil :type list)
  (decoded  (make-array 0 :element-type 'fixnum :adjustable t :fill-pointer 0)
            :type (array fixnum (*))))

(defun make-chat-session (ctx &key template system-prompt)
  "Create a stateful chat session that incrementally reuses the KV cache.
CTX must be a LLAMA-CONTEXT created with sufficient :N-CTX for the expected
conversation length.
TEMPLATE overrides the model's embedded chat template.
SYSTEM-PROMPT, when provided, seeds the conversation with an initial system
message (role \"system\").

The session borrows CTX — CTX must outlive the returned session object.
Signals INPUT-VALIDATION-ERROR if CTX is not a LLAMA-CONTEXT."
  (unless (llama-context-p ctx)
    (error 'input-validation-error
           :function-name 'make-chat-session :argument :ctx :value ctx
           :reason "ctx must be a LLAMA-CONTEXT"))
  ;; Start with a clean slate so DECODED correctly reflects an empty cache.
  (clear-kv-cache ctx)
  (let* ((raw-model (%llama:get-model (llama-context-pointer ctx)))
         (model (%make-llama-model :pointer raw-model))
         (messages (when system-prompt
                     (list (list :role "system" :content system-prompt)))))
    (%make-chat-session :context ctx :model model :template template
                        :messages messages)))

;;; --- Private helpers -------------------------------------------------------

(defun %common-prefix-length (a b)
  "Return the number of leading elements that are EQL in fixnum vectors A and B."
  (declare (type (array fixnum (*)) a b)
           (optimize (speed 3)))
  (let ((lim (min (length a) (length b))))
    (declare (type fixnum lim))
    (dotimes (i lim lim)
      (declare (type fixnum i))
      (unless (= (aref a i) (aref b i))
        (return i)))))

(defun %strip-keys (plist keys)
  "Return PLIST with any key/value pairs whose key is a member of KEYS removed."
  (loop for (k v) on plist by #'cddr
        unless (member k keys)
          nconc (list k v)))

;;; --- Public operations -----------------------------------------------------

(defun chat-session-send (session content &rest generate-keys)
  "Append a user turn with CONTENT to SESSION and generate a reply.
Returns two values: the reply string and a stop reason (:eog, :length, or
:callback).  Only the *new* tokens are decoded each turn — the KV cache is
reused for the already-decoded prefix, keeping prefill cost constant per turn.

MESSAGES is the source of truth: if the caller has modified CHAT-SESSION-MESSAGES
since the last call (appending, truncating, or editing turns), SEND reconciles
the KV cache automatically — it re-renders the full prompt, finds the common
prefix with what is cached, evicts any stale suffix, and decodes only the delta.

Additional keyword arguments are forwarded to GENERATE (e.g. :max-tokens,
:temp, :seed, :greedy, :sampler).  :PROMPT-TOKENS and :RESET-CONTEXT are
stripped if supplied — chat-session manages those internally.

Signals INPUT-VALIDATION-ERROR if CONTENT is not a non-empty string.
Signals DECODE-ERROR if the context runs out of space (n-ctx exceeded)."
  (unless (stringp content)
    (error 'input-validation-error
           :function-name 'chat-session-send :argument :content :value content
           :reason "content must be a string"))
  (when (zerop (length content))
    (error 'input-validation-error
           :function-name 'chat-session-send :argument :content :value content
           :reason "content must be non-empty"))
  (let* ((ctx      (chat-session-context  session))
         (model    (chat-session-model    session))
         (template (chat-session-template session))
         (decoded  (chat-session-decoded  session)))
    ;; 1. Record user turn (tentatively; rolled back on error via saved-msgs).
    (let ((saved-msgs (chat-session-messages session)))
      (setf (chat-session-messages session)
            (nconc (copy-list saved-msgs)
                   (list (list :role "user" :content content))))
      ;; 2. Render the full prompt including the assistant-prefix so the model
      ;;    continues into an assistant turn.
      (let* ((rendered (tokenize-chat model (chat-session-messages session)
                                      :template template
                                      :add-assistant-prefix t))
             ;; 3. Find the first divergence between what is cached and the
             ;;    newly rendered prompt.
             (k (the fixnum (%common-prefix-length decoded rendered))))
        ;; 4. Evict any stale cached suffix.  In normal multi-turn usage the
        ;;    cache is always a proper prefix of the new render (templates are
        ;;    stable), so this guard is a safety net.
        (when (< k (length decoded))
          (kv-cache-seq-rm ctx 0 k -1)
          (setf (fill-pointer decoded) k))
        ;; 5. Build the delta: new tokens not yet in the cache.
        (let* ((delta-len (- (length rendered) k))
               (delta (make-array delta-len :element-type 'fixnum)))
          (declare (type fixnum delta-len)
                   (type (simple-array fixnum (*)) delta))
          (dotimes (i delta-len)
            (declare (type fixnum i))
            (setf (aref delta i) (aref rendered (the fixnum (+ k i)))))
          ;; 6. Decode only the delta into the KV cache, then sample the reply.
          ;;    PREFILL handles the decode; GENERATE is called with empty
          ;;    :PROMPT-TOKENS so it skips re-decode and goes straight to sampling.
          (prefill ctx delta)
          (let ((begin-fn (getf (getf generate-keys :speculative-fns) :begin-fn)))
            (when begin-fn
              (funcall begin-fn 0 rendered)))
          (let* ((spec-fns (getf generate-keys :speculative-fns))
                 (safe-keys (%strip-keys generate-keys
                                         '(:prompt-tokens :reset-context
                                           :speculative-fns)))
                 (reply-keys (list* :prompt-tokens (make-array 0 :element-type 'fixnum)
                                    :reset-context nil
                                    :speculative-fns spec-fns
                                    safe-keys)))
            (handler-case
                (multiple-value-bind (reply stop-reason reply-tokens)
                    (apply #'generate ctx nil reply-keys)
                  ;; 7. Restore DECODED invariant: rendered ++ reply-tokens.
                  (loop for tok across delta
                        do (vector-push-extend tok decoded))
                  (loop for tok across reply-tokens
                        do (vector-push-extend tok decoded))
                  ;; 8. Record assistant turn.
                  (setf (chat-session-messages session)
                        (nconc (chat-session-messages session)
                               (list (list :role "assistant" :content reply))))
                  (values reply stop-reason))
              (error (e)
                ;; generate failed (e.g. decode-error from context exhaustion).
                ;; Evict the partially-decoded delta + any sampled tokens and
                ;; restore the session to its pre-call state so the caller can
                ;; handle the error (reset, shrink, etc.) and retry safely.
                (kv-cache-seq-rm ctx 0 k -1)
                (setf (fill-pointer decoded) k)
                (setf (chat-session-messages session) saved-msgs)
                (error e)))))))))

(defun chat-session-reset (session &key keep-system)
  "Clear the KV cache and reset SESSION to an empty conversation.
When KEEP-SYSTEM is true and the first message has role \"system\",
that system message is retained and the cache is seeded with it on the next
turn.
Returns NIL."
  (clear-kv-cache (chat-session-context session))
  (setf (fill-pointer (chat-session-decoded session)) 0)
  (setf (chat-session-messages session)
        (when keep-system
          (let ((first (first (chat-session-messages session))))
            (when (and first (string= (getf first :role) "system"))
              (list first)))))
  nil)
