(in-package #:cl-llama-cpp/common/reasoning-budget)

(cffi:defcfun ("llama_extras_reasoning_budget_init" %reasoning-budget-init)
    :pointer
  (vocab :pointer)
  (start-tokens :pointer) (n-start :int32)
  (end-tokens :pointer)   (n-end :int32)
  (forced-tokens :pointer) (n-forced :int32)
  (budget :int32)
  (initial-state :int))

(cffi:defcfun ("llama_extras_reasoning_budget_get_state" %reasoning-budget-get-state)
    :int
  (smpl :pointer))

(cffi:defcfun ("llama_extras_reasoning_budget_force" %reasoning-budget-force)
    :int
  (smpl :pointer))
