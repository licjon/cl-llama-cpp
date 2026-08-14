(in-package #:cl-llama-cpp/common/reasoning-budget)

(defparameter +reasoning-budget-states+
  #(:idle :counting :forcing :waiting-utf8 :done))

(define-condition reasoning-budget-init-error (cl-llama-cpp:llama-error)
  ((message :initarg :message :reader reasoning-budget-init-error-message))
  (:report (lambda (c s)
             (format s "Reasoning budget init failed: ~A"
                     (reasoning-budget-init-error-message c)))))

(defun %state-to-keyword (state-int)
  (if (and (>= state-int 0) (< state-int (length +reasoning-budget-states+)))
      (aref +reasoning-budget-states+ state-int)
      :idle))

(defun %keyword-to-state (keyword)
  (ecase keyword
    (:idle         0)
    (:counting     1)
    (:forcing      2)
    (:waiting-utf8 3)
    (:done         4)))

(defun %marshal-tokens (token-seq body-fn)
  (let ((tokens (coerce token-seq 'vector)))
    (let ((n (length tokens)))
      (if (zerop n)
          (funcall body-fn (cffi:null-pointer) 0)
          (cffi:with-foreign-object (buf :int32 n)
            (loop for i below n
                  do (setf (cffi:mem-aref buf :int32 i) (aref tokens i)))
            (funcall body-fn buf n))))))

(defun make-reasoning-budget-sampler (model start-tokens end-tokens
                                      forced-tokens budget
                                      &key (initial-state :idle))
  "Create a reasoning budget sampler that limits tokens in a thinking block.
MODEL provides the vocabulary for UTF-8 boundary detection.
START-TOKENS, END-TOKENS, and FORCED-TOKENS are sequences of token IDs.
BUDGET is the maximum number of tokens allowed in the reasoning block.
INITIAL-STATE is one of :IDLE, :COUNTING, :FORCING, :WAITING-UTF8, :DONE.
Returns a CL-LLAMA-CPP:LLAMA-SAMPLER suitable for adding to a sampler chain."
  (let ((vocab (%llama:model-get-vocab (cl-llama-cpp:llama-model-pointer model)))
        (state-int (%keyword-to-state initial-state)))
    (%marshal-tokens start-tokens
      (lambda (start-buf n-start)
        (%marshal-tokens end-tokens
          (lambda (end-buf n-end)
            (%marshal-tokens forced-tokens
              (lambda (forced-buf n-forced)
                (let ((ptr (%reasoning-budget-init
                            vocab start-buf n-start
                            end-buf n-end forced-buf n-forced
                            budget state-int)))
                  (when (cffi:null-pointer-p ptr)
                    (error 'reasoning-budget-init-error
                           :message "shim returned NULL"))
                  (cl-llama-cpp::%make-llama-sampler :pointer ptr))))))))))

(defun reasoning-budget-state (sampler)
  "Query the state of a reasoning budget SAMPLER.
Returns one of :IDLE, :COUNTING, :FORCING, :WAITING-UTF8, :DONE."
  (%state-to-keyword
   (%reasoning-budget-get-state
    (cl-llama-cpp::llama-sampler-pointer sampler))))

(defun reasoning-budget-force (sampler)
  "Force transition from :COUNTING to :FORCING state.
Returns T if the transition occurred, NIL otherwise."
  (not (zerop
        (%reasoning-budget-force
         (cl-llama-cpp::llama-sampler-pointer sampler)))))
