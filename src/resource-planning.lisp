(in-package #:cl-llama-cpp)

;;; Resource planning & configuration validation

(defun ggml-type-bytes (type)
  "Return bytes per element for a ggml-type keyword as a rational."
  (case type
    (:f32     4)
    (:f16     2)
    (:bf16    2)
    (:f64     8)
    (:i8      1)
    (:i16     2)
    (:i32     4)
    (:i64     8)
    (:q4-0    (/ 18 32))
    (:q4-1    (/ 20 32))
    (:q5-0    (/ 22 32))
    (:q5-1    (/ 24 32))
    (:q8-0    (/ 34 32))
    (:q8-1    (/ 36 32))
    (:q2-k    (/ 84 256))
    (:q3-k    (/ 110 256))
    (:q4-k    (/ 144 256))
    (:q5-k    (/ 176 256))
    (:q6-k    (/ 210 256))
    (:q8-k    (/ 292 256))
    (:iq2-xxs (/ 66 256))
    (:iq2-xs  (/ 74 256))
    (:iq3-xxs (/ 98 256))
    (:iq1-s   (/ 50 256))
    (:iq4-nl  (/ 18 32))
    (:iq3-s   (/ 110 256))
    (:iq2-s   (/ 82 256))
    (:iq4-xs  (/ 36 32))
    (:iq1-m   (/ 56 256))
    (otherwise 2)))

(defun %format-bytes (bytes)
  "Format BYTES as a human-readable string with appropriate unit."
  (cond
    ((>= bytes (* 1024 1024 1024))
     (format nil "~,1F GiB" (/ bytes (* 1024.0d0 1024 1024))))
    ((>= bytes (* 1024 1024))
     (format nil "~,1F MiB" (/ bytes (* 1024.0d0 1024))))
    (t (format nil "~,1F KiB" (/ bytes 1024.0d0)))))

(defun estimate-memory (model &key n-ctx type-k type-v)
  "Estimate memory requirements for MODEL with the given context parameters.
Returns a plist with :MODEL-SIZE, :KV-CACHE, :COMPUTE, and :TOTAL (all in bytes).
N-CTX defaults to the model's training context length.
TYPE-K and TYPE-V default to :F16."
  (with-llama-compatible-fp-environment
    (let* ((n-ctx (or n-ctx (%llama:model-n-ctx-train model)))
           (type-k (or type-k :f16))
           (type-v (or type-v :f16))
           (n-layers (%llama:model-n-layer model))
           (n-embd (%llama:model-n-embd model))
           ;; n-head/n-head-kv abort in C when n_layer_all == 0 (vocab-only models)
           (n-heads (if (zerop n-layers) 0 (%llama:model-n-head model)))
           (n-kv-heads (if (zerop n-layers) 0 (%llama:model-n-head-kv model)))
           (head-dim (if (zerop n-heads) 0 (/ n-embd n-heads)))
           (model-size (%llama:model-size model))
           (k-bytes (* n-ctx n-layers n-kv-heads head-dim
                       (ggml-type-bytes type-k)))
           (v-bytes (* n-ctx n-layers n-kv-heads head-dim
                       (ggml-type-bytes type-v)))
           (kv-cache (ceiling (+ k-bytes v-bytes)))
           (compute (ceiling (* n-embd 4 512))))
      (list :model-size model-size
            :kv-cache kv-cache
            :compute compute
            :total (+ model-size kv-cache compute)))))

(defun explain-memory-usage (model &key n-ctx type-k type-v
                                        (stream *standard-output*))
  "Print a human-readable memory breakdown for MODEL with the given parameters.
Returns NIL."
  (with-llama-compatible-fp-environment
    (let* ((n-ctx (or n-ctx (%llama:model-n-ctx-train model)))
           (type-k (or type-k :f16))
           (type-v (or type-v :f16))
           (estimate (estimate-memory model :n-ctx n-ctx
                                            :type-k type-k :type-v type-v))
           (desc (read-model-buffer-string model #'%llama:model-desc)))
      (format stream "~&Model: ~A~%~%" desc)
      (format stream "Estimated Usage~%")
      (format stream "---------------~%")
      (format stream "Model Weights:    ~A~%" (%format-bytes (getf estimate :model-size)))
      (format stream "KV Cache:         ~A  (n-ctx=~D, type-k=~A, type-v=~A)~%"
              (%format-bytes (getf estimate :kv-cache)) n-ctx type-k type-v)
      (format stream "Compute Buffers:  ~A~%" (%format-bytes (getf estimate :compute)))
      (format stream "~%")
      (format stream "Total Estimated:  ~A~%" (%format-bytes (getf estimate :total)))
      (finish-output stream)
      nil)))

(defun validate-configuration (model &key n-ctx type-k type-v
                                          n-gpu-layers vram-budget)
  "Validate whether a configuration is likely to succeed.
Returns a plist (:STATUS :SAFE/:UNSAFE/:UNKNOWN :REASON string).
Without VRAM-BUDGET, status is :UNKNOWN. When N-GPU-LAYERS is supplied,
only the proportional GPU share of model weights plus KV cache is checked
against VRAM-BUDGET."
  (let* ((estimate (estimate-memory model :n-ctx n-ctx :type-k type-k :type-v type-v))
         (model-size (getf estimate :model-size))
         (kv-cache (getf estimate :kv-cache))
         (compute (getf estimate :compute)))
    (if (null vram-budget)
        (list :status :unknown
              :reason "No VRAM budget supplied; cannot determine feasibility.")
        (with-llama-compatible-fp-environment
          (let* ((n-layers (%llama:model-n-layer model))
                 (gpu-layers (if n-gpu-layers
                                 (min n-gpu-layers n-layers)
                                 n-layers))
                 (gpu-weight-share (if (zerop n-layers)
                                       model-size
                                       (ceiling (* model-size (/ gpu-layers n-layers)))))
                 (gpu-total (+ gpu-weight-share kv-cache compute)))
            (if (<= gpu-total vram-budget)
                (list :status :safe
                      :reason (format nil "Estimated GPU usage ~A fits within ~A budget."
                                      (%format-bytes gpu-total)
                                      (%format-bytes vram-budget)))
                (list :status :unsafe
                      :reason (format nil "Estimated GPU usage ~A exceeds ~A budget by ~A."
                                      (%format-bytes gpu-total)
                                      (%format-bytes vram-budget)
                                      (%format-bytes (- gpu-total vram-budget))))))))))

(defun feasibility-report (model &key n-ctx type-k type-v n-gpu-layers vram-budget
                                      (stream *standard-output*))
  "Print a feasibility report for MODEL with the given parameters.
Returns a plist with the memory estimate and validation status."
  (with-llama-compatible-fp-environment
    (let* ((n-ctx (or n-ctx (%llama:model-n-ctx-train model)))
           (type-k (or type-k :f16))
           (type-v (or type-v :f16))
           (estimate (estimate-memory model :n-ctx n-ctx
                                            :type-k type-k :type-v type-v))
           (validation (validate-configuration model :n-ctx n-ctx
                                                     :type-k type-k :type-v type-v
                                                     :n-gpu-layers n-gpu-layers
                                                     :vram-budget vram-budget))
           (desc (read-model-buffer-string model #'%llama:model-desc))
           (status (getf validation :status))
           (reason (getf validation :reason)))
      (format stream "~&Model: ~A~%~%" desc)
      (format stream "Estimated Usage~%")
      (format stream "---------------~%")
      (format stream "Model Weights:    ~A~%" (%format-bytes (getf estimate :model-size)))
      (format stream "KV Cache:         ~A~%" (%format-bytes (getf estimate :kv-cache)))
      (format stream "Compute Buffers:  ~A~%~%" (%format-bytes (getf estimate :compute)))
      (format stream "Total Estimated:  ~A~%~%" (%format-bytes (getf estimate :total)))
      (format stream "Status: ~A~%" (string-upcase (symbol-name status)))
      (when reason
        (format stream "Reason: ~A~%" reason))
      (finish-output stream)
      (append estimate validation))))

(defun suggest-configuration (model &key n-ctx n-gpu-layers vram-budget)
  "Suggest a configuration that fits within VRAM-BUDGET.
Returns a plist (:N-CTX n :N-GPU-LAYERS n) or NIL if no viable configuration found.
Reduces N-GPU-LAYERS first, then halves N-CTX until the estimate fits."
  (unless vram-budget
    (return-from suggest-configuration nil))
  (with-llama-compatible-fp-environment
    (let* ((n-layers (%llama:model-n-layer model))
           (n-ctx (or n-ctx (%llama:model-n-ctx-train model)))
           (n-gpu-layers (if n-gpu-layers (min n-gpu-layers n-layers) n-layers)))
      (labels ((fits-p (ctx gpu-layers)
                 (let ((v (validate-configuration model :n-ctx ctx
                                                        :n-gpu-layers gpu-layers
                                                        :vram-budget vram-budget)))
                   (eq :safe (getf v :status)))))
        (when (fits-p n-ctx n-gpu-layers)
          (return-from suggest-configuration
            (list :n-ctx n-ctx :n-gpu-layers n-gpu-layers)))
        (loop for gl from (1- n-gpu-layers) downto 0
              when (fits-p n-ctx gl)
              do (return-from suggest-configuration
                   (list :n-ctx n-ctx :n-gpu-layers gl)))
        (loop for ctx = (ash n-ctx -1) then (ash ctx -1)
              while (>= ctx 128)
              do (loop for gl from n-gpu-layers downto 0
                       when (fits-p ctx gl)
                       do (return-from suggest-configuration
                            (list :n-ctx ctx :n-gpu-layers gl))))
        nil))))

(defun %validate-context-params (model ctx-params validation vram-budget)
  "Internal: run validation checks before context creation."
  (when (and validation (not (eq validation :off)))
    (let* ((n-ctx (getf ctx-params '%llama:n-ctx))
           (type-k (getf ctx-params '%llama:type-k))
           (type-v (getf ctx-params '%llama:type-v))
           (estimate (estimate-memory model :n-ctx n-ctx
                                            :type-k type-k :type-v type-v)))
      (when vram-budget
        (let ((total (getf estimate :total)))
          (when (> total vram-budget)
            (let ((reason (format nil "Estimated ~A exceeds ~A budget."
                                  (%format-bytes total)
                                  (%format-bytes vram-budget))))
              (ecase validation
                (:warn (warn 'configuration-unsafe-warning :reason reason))
                (:error (error 'configuration-unsafe-error
                               :reason reason))))))))))
