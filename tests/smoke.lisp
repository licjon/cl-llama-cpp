(defpackage #:cl-llama-cpp/tests/smoke
  (:use #:cl #:rove))

(in-package #:cl-llama-cpp/tests/smoke)

(deftest bindings-package-exists
  (testing "%llama package exists and has symbols"
    (let ((pkg (find-package :%llama)))
      (ok pkg "%llama package exists")
      (let ((count 0))
        (do-symbols (s pkg) (incf count))
        (ok (> count 50) (format nil "%llama has ~d symbols (expected >50)" count))))))

(deftest model-default-params
  (testing "llama_model_default_params returns without error"
    (cl-llama-cpp:with-fp-traps-masked
      ;; CLAW generates SRET (struct-return) convention: allocate result buffer
      ;; and pass as pointer; use foreign-funcall to bypass type translation.
      (let ((buf (cffi:foreign-alloc :uint8 :count 72)))
        (unwind-protect
             (progn
               (cffi:foreign-funcall "llama_model_default_params" :pointer buf :void)
               (ok (not (cffi:null-pointer-p buf))
                   "model-default-params returned non-nil"))
          (cffi:foreign-free buf))))))

(deftest context-default-params
  (testing "llama_context_default_params returns without error"
    (cl-llama-cpp:with-fp-traps-masked
      ;; CLAW generates SRET (struct-return) convention: allocate result buffer
      ;; and pass as pointer; use foreign-funcall to bypass type translation.
      (let ((buf (cffi:foreign-alloc :uint8 :count 120)))
        (unwind-protect
             (progn
               (cffi:foreign-funcall "llama_context_default_params" :pointer buf :void)
               (ok (not (cffi:null-pointer-p buf))
                   "context-default-params returned non-nil"))
          (cffi:foreign-free buf))))))
