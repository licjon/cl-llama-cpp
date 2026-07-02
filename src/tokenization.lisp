(in-package #:cl-llama-cpp)

(defun %utf-8-byte-length (string)
  "Return the number of bytes needed to encode STRING as UTF-8."
  (declare (optimize (speed 3)) (type string string))
  (let ((n 0))
    (declare (type fixnum n))
    (dotimes (i (length string) n)
      (declare (type fixnum i))
      (let ((code (char-code (char string i))))
        (declare (type fixnum code))
        (incf n (cond ((<= code #x7F) 1)
                      ((<= code #x7FF) 2)
                      ((<= code #xFFFF) 3)
                      (t 4)))))))

(llama-defun tokenize (model text &key (add-special t) (parse-special nil))
  "Tokenize TEXT using MODEL's vocabulary. Returns a vector of token integers.
Signals INPUT-VALIDATION-ERROR if TEXT is not a string."
  (declare (optimize (speed 3)))
  (check-type text string)
  (let* ((vocab (%llama:model-get-vocab (llama-model-pointer model)))
         (text-len (%utf-8-byte-length text))
         (add-sp (if add-special 1 0))
         (parse-sp (if parse-special 1 0))
         ;; First pass: get required token count
         (n-needed (- (%llama:tokenize vocab text text-len
                                       (cffi:null-pointer) 0
                                       add-sp parse-sp))))
    (declare (type fixnum text-len add-sp parse-sp n-needed))
    (when (<= n-needed 0)
      (error 'tokenization-error :text text))
    ;; Second pass: fill token buffer
    (cffi:with-foreign-object (buf '%llama:token n-needed)
      (let ((n-written (%llama:tokenize vocab text text-len
                                        buf n-needed
                                        add-sp parse-sp)))
        (declare (type fixnum n-written))
        (when (< n-written 0)
          (error 'tokenization-error :text text))
        (let ((result (make-array n-written :element-type 'fixnum)))
          (declare (type (simple-array fixnum (*)) result))
          (dotimes (i n-written result)
            (declare (type fixnum i))
            (setf (aref result i)
                  (cffi:mem-aref buf '%llama:token i))))))))

(llama-defun detokenize (model tokens &key (remove-special nil) (unparse-special t))
  "Detokenize a vector of TOKENS using MODEL's vocabulary. Returns a string.
Signals a TYPE-ERROR if TOKENS is not a vector."
  (declare (optimize (speed 3)))
  (check-type tokens vector)
  (let* ((vocab (%llama:model-get-vocab (llama-model-pointer model)))
         (n-tokens (length tokens))
         (remove-sp (if remove-special 1 0))
         (unparse-sp (if unparse-special 1 0)))
    (declare (type fixnum n-tokens remove-sp unparse-sp))
    ;; Copy tokens into foreign buffer
    (cffi:with-foreign-object (tok-buf '%llama:token n-tokens)
      (dotimes (i n-tokens)
        (declare (type fixnum i))
        (setf (cffi:mem-aref tok-buf '%llama:token i) (aref tokens i)))
      ;; First pass: get required text length
      (let ((n-needed (- (%llama:detokenize vocab tok-buf n-tokens
                                            (cffi:null-pointer) 0
                                            remove-sp unparse-sp))))
        (declare (type fixnum n-needed))
        (when (<= n-needed 0)
          (return-from detokenize ""))
        ;; Second pass: fill text buffer (llama API does not null-terminate)
        (cffi:with-foreign-pointer (text-buf (1+ n-needed))
          (let ((n-written (%llama:detokenize vocab tok-buf n-tokens
                                              text-buf n-needed
                                              remove-sp unparse-sp)))
            (declare (type fixnum n-written))
            (cffi:foreign-string-to-lisp text-buf :count n-written)))))))
