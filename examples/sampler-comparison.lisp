;;; Sampler strategy comparison — demonstrates the extended sampler wrappers
;;; by running the same prompt through different sampling configurations and
;;; comparing the output side-by-side.
;;;
;;; Covers: typical-p, xtc, top-n-sigma, mirostat v1/v2, repeat/frequency/
;;; presence penalties, DRY anti-repetition, logit-bias, dynamic temperature,
;;; adaptive-p, sampler-seed, and make-sampler-config (reusable configs).
;;;
;;; Setup:
;;;   (ql:quickload :cl-llama-cpp)
;;;   (load "examples/sampler-comparison.lisp")
;;;   (setf cl-llama-cpp/examples/sampler-comparison::*model-path*
;;;         "/path/to/model.gguf")
;;;   (cl-llama-cpp/examples/sampler-comparison:run)
;;;
;;; Or via environment variable:
;;;   export LLAMA_MODEL=/path/to/model.gguf

(defpackage #:cl-llama-cpp/examples/sampler-comparison
  (:use #:cl #:cl-llama-cpp)
  (:export #:run))

(in-package #:cl-llama-cpp/examples/sampler-comparison)

(defvar *model-path* (uiop:getenv "LLAMA_MODEL"))

;;; ── Helpers ──────────────────────────────────────────────────────────

(defun banner (title)
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  ~A~%" title)
  (format t "~A~2%" (make-string 64 :initial-element #\═)))

(defun section (title)
  (format t "~&~%── ~A ──~2%" title))

(defun gen (ctx prompt &rest sampler-args &key (max-tokens 96) &allow-other-keys)
  "Generate with streaming, return (values text stop-reason).
Strips leading whitespace from output for cleaner display."
  (let ((clean-args (loop for (k v) on sampler-args by #'cddr
                          unless (eq k :max-tokens)
                          append (list k v)))
        (first-token t))
    (multiple-value-bind (text stop-reason)
        (apply #'generate ctx prompt
               :max-tokens max-tokens
               :token-callback (lambda (tok)
                                 (when first-token
                                   (setf tok (string-left-trim '(#\Space #\Newline #\Tab) tok)
                                         first-token nil))
                                 (write-string tok)
                                 (force-output)
                                 t)
               clean-args)
      (terpri)
      (values text stop-reason))))

;;; ── Demo 1: Baseline vs. penalty-based repetition suppression ──────

(defun demo-penalties (ctx)
  (banner "DEMO 1: Repetition Penalties")

  (format t "When generating multiple similar passages, models tend to reuse~%")
  (format t "the same words and sentence structures.  Three penalty knobs~%")
  (format t "push toward more diverse vocabulary:~%")
  (format t "  :REPEAT-PENALTY   — penalises exact token repeats~%")
  (format t "  :FREQUENCY-PENALTY — scales with how often a token appeared~%")
  (format t "  :PRESENCE-PENALTY  — flat penalty if the token appeared at all~%")

  (let ((prompt "Describe five different types of flowers, one paragraph each:"))
    (section "1a: No penalties (baseline)")
    (format t "Prompt: ~S~%" prompt)
    (gen ctx prompt :temp 0.7 :seed 42 :max-tokens 350)

    (section "1b: With penalties")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :repeat-penalty 1.2  :frequency-penalty 0.4~%")
    (format t "  :presence-penalty 0.4  :penalty-last-n 256~%")
    (gen ctx prompt :temp 0.7 :seed 42 :max-tokens 350
         :repeat-penalty 1.2
         :frequency-penalty 0.4
         :presence-penalty 0.4
         :penalty-last-n 256))
  (format t "~%Compare: without penalties each paragraph reuses the same words~%")
  (format t "(\"delicate\", \"fragrance\", \"beauty\") and sentence patterns.~%")
  (format t "With penalties the model reaches for more varied vocabulary.~%"))

;;; ── Demo 2: DRY anti-repetition ────────────────────────────────────

(defun demo-dry (ctx)
  (banner "DEMO 2: DRY Anti-Repetition")

  (format t "DRY (Don't Repeat Yourself) penalises repeated *sequences*,~%")
  (format t "not just individual tokens.  This catches phrase-level loops~%")
  (format t "that per-token penalties miss.  :DRY-SEQ-BREAKERS defines~%")
  (format t "tokens that reset the sequence matcher (e.g. newlines).~%")

  (let ((prompt "Describe the steps to make a cup of tea, then describe the steps to make a cup of coffee."))
    (section "2a: No DRY (baseline)")
    (format t "Prompt: ~S~%" prompt)
    (gen ctx prompt :temp 0.9 :seed 77 :max-tokens 192)

    (section "2b: With DRY")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :dry-multiplier 0.8  :dry-base 1.75  :dry-allowed-length 2~%")
    (format t "  :dry-seq-breakers (newline, period, colon)~%")
    (gen ctx prompt :temp 0.9 :seed 77 :max-tokens 192
         :dry-multiplier 0.8
         :dry-base 1.75
         :dry-allowed-length 2
         :dry-penalty-last-n 256
         :dry-seq-breakers '("\n" "." ":")))
  (format t "~%Watch how the coffee steps use different phrasing from the tea steps.~%"))

;;; ── Demo 3: Mirostat — perplexity-targeted sampling ────────────────

(defun demo-mirostat (ctx)
  (banner "DEMO 3: Mirostat — Adaptive Perplexity Control")

  (format t "Mirostat replaces the top-k/top-p/temp pipeline with a single~%")
  (format t "feedback loop that targets a desired surprise level (tau).~%")
  (format t "Lower tau = more focused and predictable output.~%")
  (format t "Higher tau = more varied and creative output.~%")
  (format t "V1 and V2 differ in their adaptation algorithm; generate~%")
  (format t "handles the model reference for both automatically.~%")

  (let ((prompt "Invent a new mythological creature and describe it:"))
    (section "3a: Mirostat V2, tau=2.0 (focused)")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :mirostat-v2 t  :mirostat-tau 2.0  :mirostat-eta 0.1~%")
    (gen ctx prompt :mirostat-v2 t :mirostat-tau 2.0 :mirostat-eta 0.1
         :seed 42 :max-tokens 128)

    (section "3b: Mirostat V2, tau=10.0 (creative)")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :mirostat-v2 t  :mirostat-tau 10.0  :mirostat-eta 0.1~%")
    (gen ctx prompt :mirostat-v2 t :mirostat-tau 10.0 :mirostat-eta 0.1
         :seed 42 :max-tokens 128)

    (section "3c: Mirostat V1, tau=5.0")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :mirostat t  :mirostat-tau 5.0  :mirostat-eta 0.1~%")
    (gen ctx prompt :mirostat t :mirostat-tau 5.0 :mirostat-eta 0.1
         :seed 42 :max-tokens 128))
  (format t "~%Mirostat adapts on the fly — no manual top-k/top-p tuning needed.~%"))

;;; ── Demo 4: Filtering strategies — typical, xtc, top-n-sigma ──────

(defun demo-filters (ctx)
  (banner "DEMO 4: Token Filtering — Typical-p, XTC, Top-N-Sigma")

  (format t "Beyond top-k and top-p, several newer filtering strategies~%")
  (format t "select which tokens remain in the candidate pool before~%")
  (format t "sampling.  Each shapes text character differently.~%")

  (let ((prompt "The old house at the end of the street"))
    (section "4a: Typical-p (p=0.9)")
    (format t "Prompt: ~S~%" prompt)
    (format t "Keeps tokens whose information content is close to the~%")
    (format t "expected information — filters out both boring and bizarre.~%")
    (format t "  :typical-p 0.9  :temp 0.9~%")
    (gen ctx prompt :temp 0.9 :typical-p 0.9 :seed 42)

    (section "4b: XTC — Exclude Top Choices (prob=0.5, threshold=0.1)")
    (format t "Prompt: ~S~%" prompt)
    (format t "Randomly excludes the most probable tokens, forcing the model~%")
    (format t "to pick less predictable alternatives.  Good for creative text.~%")
    (format t "  :xtc-probability 0.5  :xtc-threshold 0.1  :temp 0.9~%")
    (gen ctx prompt :temp 0.9 :xtc-probability 0.5 :xtc-threshold 0.1 :seed 42)

    (section "4c: Top-N-Sigma (n=2.0)")
    (format t "Prompt: ~S~%" prompt)
    (format t "Keeps tokens within N standard deviations of the top logit.~%")
    (format t "Adapts pool size to the model's confidence at each step.~%")
    (format t "  :top-n-sigma 2.0  :temp 0.9~%")
    (gen ctx prompt :temp 0.9 :top-n-sigma 2.0 :seed 42))
  (format t "~%Each filter shapes the output character differently from the same prompt.~%"))

;;; ── Demo 5: Dynamic temperature and adaptive-p ─────────────────────

(defun demo-dynamic-temp (ctx)
  (banner "DEMO 5: Dynamic Temperature & Adaptive-P")

  (format t "Dynamic temperature (temp-ext) adjusts temperature based on~%")
  (format t "the entropy of the logit distribution — low entropy (confident)~%")
  (format t "keeps temp low, high entropy (uncertain) raises it.~%")
  (format t ":DYNAMIC-TEMP-RANGE sets the +/- range around :TEMP.~%")
  (format t ":DYNAMIC-TEMP-EXPONENT shapes the mapping curve.~%")

  (let ((prompt "Describe a sunset over the mountains in three sentences:"))
    (section "5a: Fixed temperature (0.7)")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :temp 0.7~%")
    (gen ctx prompt :temp 0.7 :seed 42)

    (section "5b: Dynamic temperature (base=0.7, range=0.5, exponent=1.5)")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :temp 0.7  :dynamic-temp-range 0.5  :dynamic-temp-exponent 1.5~%")
    (format t "Temperature varies between 0.2 and 1.2 based on entropy.~%")
    (gen ctx prompt :temp 0.7 :dynamic-temp-range 0.5
         :dynamic-temp-exponent 1.5 :seed 42)

    (section "5c: Adaptive-P (target=0.9, decay=0.01)")
    (format t "Prompt: ~S~%" prompt)
    (format t "Adaptive-P is a dynamic variant of top-p that adjusts the~%")
    (format t "probability threshold toward the target over time.~%")
    (format t "Note: this sampler is experimental and can destabilise~%")
    (format t "small models — tune carefully for your model size.~%")
    (format t "  :temp 0.7  :adaptive-p 0.9  :adaptive-p-decay 0.01~%")
    (gen ctx prompt :temp 0.7 :adaptive-p 0.9 :adaptive-p-decay 0.01 :seed 42))
  (format t "~%Dynamic sampling adapts to the model's confidence per-token.~%"))

;;; ── Demo 6: Logit bias ─────────────────────────────────────────────

(defun demo-logit-bias (model ctx)
  (banner "DEMO 6: Logit Bias — Steering Token Probabilities")

  (format t "Logit bias adds a fixed value to specific token logits before~%")
  (format t "sampling.  Positive values increase probability; large negative~%")
  (format t "values effectively ban tokens.  Format: alist of (token-id . bias).~%")

  (let* ((prompt "Name three colors:")
         (banned-words '("red" "Red" " red" " Red"
                         "blue" "Blue" " blue" " Blue"))
         (biases (loop for word in banned-words
                       append (map 'list (lambda (tok) (cons tok -100.0))
                                   (tokenize model word :add-special nil)))))
    (setf biases (remove-duplicates biases :key #'car))

    (section "6a: No logit bias (baseline)")
    (format t "Prompt: ~S~%" prompt)
    (gen ctx prompt :temp 0.3 :seed 42)

    (section "6b: Ban 'red' and 'blue' tokens (bias = -100)")
    (format t "Prompt: ~S~%" prompt)
    (format t "Banned ~D token IDs covering case/spacing variants of \"red\", \"blue\"~%" (length biases))
    (gen ctx prompt :temp 0.3 :seed 42 :logit-bias biases))
  (format t "~%Logit bias gives fine-grained control over which tokens can appear.~%"))

;;; ── Demo 7: Sampler seed inspection ────────────────────────────────

(defun demo-sampler-seed (ctx)
  (banner "DEMO 7: Sampler Seed — Reproducibility & Inspection")

  (format t "sampler-seed reads the current RNG seed from a sampler chain.~%")
  (format t "Useful for logging the seed that produced a given output, or~%")
  (format t "verifying reproducibility across runs.~%")

  (section "7a: Inspecting a chain's seed")
  (with-sampler-chain (chain :temp 0.8 :seed 12345)
    (format t "  Chain created with :seed 12345~%")
    (format t "  sampler-seed reports: ~A~%" (sampler-seed chain)))

  (section "7b: Reproducibility — same seed, same output")
  (let ((prompt "Once upon a time,"))
    (format t "Prompt: ~S  (seed=999)~%" prompt)
    (format t "Run 1: ")
    (gen ctx prompt :temp 0.8 :seed 999 :max-tokens 48)
    (format t "Run 2: ")
    (gen ctx prompt :temp 0.8 :seed 999 :max-tokens 48))
  (format t "~%Identical seeds produce identical output (deterministic sampling).~%")

  (section "7c: Nondeterministic sampling with :seed :random")
  (let ((prompt "Once upon a time,"))
    (format t "Prompt: ~S  (seed=:random)~%" prompt)
    (format t "Run 1: ")
    (gen ctx prompt :temp 0.8 :seed :random :max-tokens 48)
    (format t "Run 2: ")
    (gen ctx prompt :temp 0.8 :seed :random :max-tokens 48))
  (format t "~%:seed :random draws a fresh seed each call — different output every time.~%")
  (format t ":seed nil is an alias for :random.~%"))

;;; ── Demo 8: Kitchen sink — combining multiple strategies ───────────

(defun demo-combined (ctx)
  (banner "DEMO 8: Combined — Production-Quality Sampling")

  (format t "Real applications combine multiple samplers.  Here we build a~%")
  (format t "chain that uses penalties + DRY + typical-p + dynamic temp~%")
  (format t "for high-quality, non-repetitive creative writing.~%")

  (let ((prompt "Write a short story opening about a lighthouse keeper who discovers something unexpected:"))
    (section "Full configuration")
    (format t "Prompt: ~S~%" prompt)
    (format t "  :typical-p 0.95          — filter by information content~%")
    (format t "  :repeat-penalty 1.2      — discourage exact repeats~%")
    (format t "  :presence-penalty 0.2    — encourage new topics~%")
    (format t "  :dry-multiplier 0.8      — suppress phrase-level loops~%")
    (format t "  :dynamic-temp-range 0.2  — adapt temp to confidence~%")
    (format t "  :temp 0.7               — base temperature~%")
    (gen ctx prompt
         :temp 0.7
         :typical-p 0.95
         :repeat-penalty 1.2
         :presence-penalty 0.2
         :penalty-last-n 128
         :dry-multiplier 0.8
         :dry-base 1.75
         :dry-seq-breakers '("\n" "." "!" "?")
         :dynamic-temp-range 0.2
         :dynamic-temp-exponent 1.0
         :seed 42
         :max-tokens 192))
  (format t "~%Multiple samplers compose naturally through keyword arguments.~%"))

;;; ── Demo 9: make-sampler-config — reusable parameter bundles ───────

(defun demo-sampler-config (ctx)
  (banner "DEMO 9: make-sampler-config — Reusable Sampling Policies")

  (format t "make-sampler-config bundles sampler parameters into a plain plist.~%")
  (format t "Pass it as :sampler-config to generate or with-sampler-chain.~%")
  (format t "Explicit keyword arguments always override what the config supplies.~%")
  (format t "Only the keys you provide are stored — no implicit defaults.~%")

  (let ((creative (make-sampler-config
                    :temp 1.0 :top-p 0.95 :repeat-penalty 1.2
                    :frequency-penalty 0.3 :seed 42))
        (focused  (make-sampler-config
                    :temp 0.2 :top-k 40 :seed 42)))

    (section "9a: Creative config — reused across two different prompts")
    (format t "Config: :temp 1.0  :top-p 0.95  :repeat-penalty 1.2  :freq-penalty 0.3~%")
    (format t "Prompt 1:~%")
    (gen ctx "Write a whimsical opening sentence for a fantasy novel:"
         :sampler-config creative :max-tokens 64)
    (format t "Prompt 2 (same config):~%")
    (gen ctx "Describe an unusual cloud formation:"
         :sampler-config creative :max-tokens 64)

    (section "9b: Focused config — low temperature for factual output")
    (format t "Config: :temp 0.2  :top-k 40~%")
    (gen ctx "What is the capital of Japan?"
         :sampler-config focused :max-tokens 32)

    (section "9c: Explicit kwarg overrides config (temp 0.9 beats config's 0.2)")
    (format t "Same focused config, but :temp 0.9 overrides at the call site:~%")
    (gen ctx "Describe a rainy afternoon:"
         :sampler-config focused :temp 0.9 :max-tokens 64)

    (section "9d: Config works with with-sampler-chain")
    (format t "Build a chain from the creative config, then pass :sampler to generate:~%")
    (with-sampler-chain (chain :sampler-config creative)
      (gen ctx "Once upon a midnight dreary,"
           :sampler chain :max-tokens 64)))

  (format t "~%The config object is a plain plist — getf works on it directly.~%")
  (format t "Keyword API calls without :sampler-config are completely unaffected.~%"))

;;; ── Entry point ────────────────────────────────────────────────────

(defun run ()
  "Run all sampler comparison demos."
  (unless *model-path*
    (error "Set *model-path* or export LLAMA_MODEL before calling run."))
  (format t "~&Loading model: ~A~%" *model-path*)
  (with-model (model *model-path* :n-gpu-layers 99)
    (with-context (ctx model :n-ctx 2048)
      (demo-penalties ctx)
      (demo-dry ctx)
      (demo-mirostat ctx)
      (demo-filters ctx)
      (demo-dynamic-temp ctx)
      (demo-logit-bias model ctx)
      (demo-sampler-seed ctx)
      (demo-combined ctx)
      (demo-sampler-config ctx)))
  (format t "~&~%~A~%" (make-string 64 :initial-element #\═))
  (format t "  All demos complete.~%")
  (format t "~A~%" (make-string 64 :initial-element #\═))
  (values))
