(in-package #:cl-llama-cpp)

;;; Session state save/load wrappers

(defun save-session (ctx path &optional tokens)
  "Save full context state to a session file at PATH.
TOKENS is an optional vector of token integers to store alongside the state."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (n-tokens (if tokens (length tokens) 0))
           (path-str (namestring path)))
      (if (zerop n-tokens)
          (let ((rc (%llama:state-save-file ctx-ptr path-str (cffi:null-pointer) 0)))
            (when (zerop rc)
              (error 'session-save-error :path path-str))
            nil)
          (let ((tok-buf (cffi:foreign-alloc '%llama:token :count n-tokens)))
            (unwind-protect
                (progn
                  (dotimes (i n-tokens)
                    (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
                  (let ((rc (%llama:state-save-file ctx-ptr path-str tok-buf n-tokens)))
                    (when (zerop rc)
                      (error 'session-save-error :path path-str))
                    nil))
              (cffi:foreign-free tok-buf)))))))

(defun load-session (ctx path)
  "Load context state from a session file at PATH.
Returns a vector of cached token integers that were stored with the state."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (path-str (namestring path))
           (capacity (%llama:n-ctx ctx-ptr))
           (tok-buf (cffi:foreign-alloc '%llama:token :count capacity)))
      (unwind-protect
          (cffi:with-foreign-object (count-out '%llama:size-t)
            (let ((rc (%llama:state-load-file ctx-ptr path-str tok-buf capacity count-out)))
              (when (zerop rc)
                (error 'session-load-error :path path-str))
              (let* ((n-tokens (cffi:mem-ref count-out '%llama:size-t))
                     (result (make-array n-tokens :element-type 'fixnum)))
                (dotimes (i n-tokens result)
                  (setf (aref result i)
                        (cffi:mem-aref tok-buf '%llama:token i))))))
        (cffi:foreign-free tok-buf)))))

(defun save-session-seq (ctx path seq-id &optional tokens)
  "Save a single sequence's state to a file at PATH.
TOKENS is an optional vector of token integers to store alongside the state."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (n-tokens (if tokens (length tokens) 0))
           (path-str (namestring path)))
      (if (zerop n-tokens)
          (let ((rc (%llama:state-seq-save-file
                     ctx-ptr path-str seq-id (cffi:null-pointer) 0)))
            (when (zerop rc)
              (error 'session-save-error :path path-str))
            nil)
          (let ((tok-buf (cffi:foreign-alloc '%llama:token :count n-tokens)))
            (unwind-protect
                (progn
                  (dotimes (i n-tokens)
                    (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
                  (let ((rc (%llama:state-seq-save-file
                             ctx-ptr path-str seq-id tok-buf n-tokens)))
                    (when (zerop rc)
                      (error 'session-save-error :path path-str))
                    nil))
              (cffi:foreign-free tok-buf)))))))

(defun load-session-seq (ctx path seq-id)
  "Load a single sequence's state from a file at PATH.
Returns a vector of cached token integers that were stored with the state."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (path-str (namestring path))
           (capacity (%llama:n-ctx ctx-ptr))
           (tok-buf (cffi:foreign-alloc '%llama:token :count capacity)))
      (unwind-protect
          (cffi:with-foreign-object (count-out '%llama:size-t)
            (let ((rc (%llama:state-seq-load-file
                       ctx-ptr path-str seq-id tok-buf capacity count-out)))
              (when (zerop rc)
                (error 'session-load-error :path path-str))
              (let* ((n-tokens (cffi:mem-ref count-out '%llama:size-t))
                     (result (make-array n-tokens :element-type 'fixnum)))
                (dotimes (i n-tokens result)
                  (setf (aref result i)
                        (cffi:mem-aref tok-buf '%llama:token i))))))
        (cffi:foreign-free tok-buf)))))

(defun save-state (ctx)
  "Serialize full context state to a Lisp octet vector."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (size (%llama:state-get-size ctx-ptr)))
      (when (zerop size)
        (return-from save-state
          (make-array 0 :element-type '(unsigned-byte 8))))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (let* ((written (%llama:state-get-data ctx-ptr buf size))
                   (result (make-array written :element-type '(unsigned-byte 8))))
              (dotimes (i written result)
                (setf (aref result i) (cffi:mem-aref buf :uint8 i))))
          (cffi:foreign-free buf))))))

(defun load-state (ctx state-bytes)
  "Restore context state from a Lisp octet vector STATE-BYTES.
Returns the number of bytes consumed."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (size (length state-bytes)))
      (when (zerop size)
        (return-from load-state 0))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (progn
              (dotimes (i size)
                (setf (cffi:mem-aref buf :uint8 i) (aref state-bytes i)))
              (%llama:state-set-data ctx-ptr buf size))
          (cffi:foreign-free buf))))))

(defun save-state-seq (ctx seq-id &key flags)
  "Serialize one sequence's state to a Lisp octet vector.
When FLAGS is provided, uses the extended variant with llama_state_seq_flags."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (size (if flags
                     (%llama:state-seq-get-size-ext ctx-ptr seq-id flags)
                     (%llama:state-seq-get-size ctx-ptr seq-id))))
      (when (zerop size)
        (return-from save-state-seq
          (make-array 0 :element-type '(unsigned-byte 8))))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (let* ((written (if flags
                                (%llama:state-seq-get-data-ext ctx-ptr buf size seq-id flags)
                                (%llama:state-seq-get-data ctx-ptr buf size seq-id)))
                   (result (make-array written :element-type '(unsigned-byte 8))))
              (dotimes (i written result)
                (setf (aref result i) (cffi:mem-aref buf :uint8 i))))
          (cffi:foreign-free buf))))))

(defun load-state-seq (ctx seq-id state-bytes &key flags)
  "Restore one sequence's state from a Lisp octet vector STATE-BYTES.
When FLAGS is provided, uses the extended variant with llama_state_seq_flags.
Returns the number of bytes consumed."
  (with-llama-compatible-fp-environment
    (let* ((ctx-ptr (llama-context-pointer ctx))
           (size (length state-bytes)))
      (when (zerop size)
        (return-from load-state-seq 0))
      (let ((buf (cffi:foreign-alloc :uint8 :count size)))
        (unwind-protect
            (progn
              (dotimes (i size)
                (setf (cffi:mem-aref buf :uint8 i) (aref state-bytes i)))
              (if flags
                  (%llama:state-seq-set-data-ext ctx-ptr buf size seq-id flags)
                  (%llama:state-seq-set-data ctx-ptr buf size seq-id)))
          (cffi:foreign-free buf))))))
