(in-package #:cl-llama-cpp)

;;; LoRA adapter wrappers

(defmacro with-lora ((var model path) &body body)
  "Load a LoRA adapter from PATH for MODEL, bind it to VAR, execute BODY, free the adapter.
If loading fails, SKIP-LORA binds VAR to NIL and continues; USE-DIFFERENT-PATH retries."
  (let ((adapter-ptr (gensym "ADAPTER"))
        (model-val (gensym "MODEL"))
        (path-val (gensym "PATH")))
    `(progn
       (ensure-backend)
       (with-llama-compatible-fp-environment
         (let* ((,model-val ,model)
                (,path-val ,path)
                (,adapter-ptr (%llama:adapter-lora-init
                               (llama-model-pointer ,model-val) ,path-val)))
           (when (cffi:null-pointer-p ,adapter-ptr)
             (setf ,adapter-ptr
                   (restart-case (error 'lora-load-error :path ,path-val)
                     (use-different-path (new-path)
                       :report "Retry loading LoRA from a different path"
                       :interactive (lambda ()
                                      (format *query-io* "LoRA path: ")
                                      (list (read-line *query-io*)))
                       (%llama:adapter-lora-init
                        (llama-model-pointer ,model-val) new-path))
                     (skip-lora ()
                       :report "Continue without loading a LoRA adapter"
                       nil))))
           (let ((,var ,adapter-ptr))
             (unwind-protect
                  (progn ,@body)
               (when (and ,adapter-ptr
                          (not (cffi:null-pointer-p ,adapter-ptr)))
                 (%llama:adapter-lora-free ,adapter-ptr)))))))))

(llama-defun apply-lora (ctx adapter &key (scale 1.0))
  "Set the active LoRA adapter on CTX to ADAPTER with the given SCALE factor.
Replaces any previously applied adapters — calling this twice does not
compose; only the last call's adapter remains active.
Returns NIL on success, signals LORA-APPLY-ERROR on failure."
  (let ((scale-f (coerce scale 'single-float)))
    (unless (<= most-negative-single-float scale-f most-positive-single-float)
      (error 'type-error :datum scale :expected-type 'single-float))
    (cffi:with-foreign-objects ((adapters-buf :pointer 1)
                                (scales-buf :float 1))
      (setf (cffi:mem-aref adapters-buf :pointer 0) adapter)
      (setf (cffi:mem-aref scales-buf :float 0) scale-f)
      (let ((rc (%llama:set-adapters-lora (llama-context-pointer ctx) adapters-buf 1 scales-buf)))
        (unless (zerop rc)
          (restart-case (error 'lora-apply-error :code rc)
            (use-different-scale (s)
              :report "Retry applying the adapter with a different scale"
              :interactive (lambda ()
                             (format *query-io* "Scale (float): ")
                             (list (read *query-io*)))
              (apply-lora ctx adapter :scale s))
            (skip-apply ()
              :report "Continue without applying the LoRA adapter"
              nil)))
        nil))))

(defun read-adapter-meta-string (adapter index reader-fn)
  "Read a metadata string from ADAPTER at INDEX using READER-FN.
READER-FN is called as (funcall reader-fn adapter index buf buf-size)."
  (let ((buf-size 256))
    (cffi:with-foreign-pointer (buf buf-size)
      (let ((n (funcall reader-fn adapter index buf buf-size)))
        (when (>= n buf-size)
          (let ((retry-size (1+ n)))
            (cffi:with-foreign-pointer (buf2 retry-size)
              (let ((n2 (funcall reader-fn adapter index buf2 retry-size)))
                (return-from read-adapter-meta-string
                  (cffi:foreign-string-to-lisp buf2 :count (max 0 n2)))))))
        (cffi:foreign-string-to-lisp buf :count (max 0 n))))))

(llama-defun lora-metadata (adapter)
  "Return metadata from ADAPTER as an alist of (key . value) string pairs."
  (let ((count (%llama:adapter-meta-count adapter)))
    (loop for i from 0 below count
          collect (cons
                   (read-adapter-meta-string
                    adapter i #'%llama:adapter-meta-key-by-index)
                   (read-adapter-meta-string
                    adapter i #'%llama:adapter-meta-val-str-by-index)))))
