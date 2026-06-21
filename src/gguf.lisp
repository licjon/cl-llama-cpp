(in-package #:cl-llama-cpp)

;;; GGUF file inspection API

(defmacro with-gguf ((var path &key no-alloc) &body body)
  "Open a GGUF file at PATH, bind the context to VAR, execute BODY, free on exit.
:NO-ALLOC T skips ggml tensor allocation (fast metadata-only reads).
Signals GGUF-LOAD-ERROR if the file cannot be opened."
  (let ((ptr (gensym "PTR"))
        (path-val (gensym "PATH")))
    `(with-llama-compatible-fp-environment
       (let* ((,path-val ,path)
              (,ptr (%llama:gguf-init-from-file
                     ,path-val
                     (list '%llama:no-alloc (%bool->c ,no-alloc)
                           '%llama:ctx (cffi:null-pointer)))))
         (when (cffi:null-pointer-p ,ptr)
           (error 'gguf-load-error :path ,path-val))
         (let ((,var (%make-gguf-context :pointer ,ptr)))
           (unwind-protect
                (progn ,@body)
             (%llama:gguf-free ,ptr)))))))

;;; File-level metadata

(defun gguf-version (gguf)
  "Return the GGUF format version as an integer."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-version (gguf-context-pointer gguf))))

(defun gguf-alignment (gguf)
  "Return the alignment value (bytes) stored in GGUF."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-alignment (gguf-context-pointer gguf))))

(defun gguf-data-offset (gguf)
  "Return the byte offset of the tensor data section within the GGUF file."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-data-offset (gguf-context-pointer gguf))))

;;; KV metadata — discovery

(defun gguf-n-kv (gguf)
  "Return the number of KV metadata entries in GGUF."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-n-kv (gguf-context-pointer gguf))))

(defun gguf-find-key (gguf key)
  "Return the integer index of metadata KEY in GGUF, or NIL if not present."
  (with-llama-compatible-fp-environment
    (let ((id (%llama:gguf-find-key (gguf-context-pointer gguf) key)))
      (unless (minusp id) id))))

(defun gguf-key (gguf key-id)
  "Return the key name string at index KEY-ID in GGUF."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-key (gguf-context-pointer gguf) key-id)))

(defun gguf-kv-type (gguf key-id)
  "Return the gguf-type keyword for KV entry KEY-ID.
Values include :UINT8 :INT8 :UINT16 :INT16 :UINT32 :INT32
:FLOAT32 :BOOL :STRING :ARRAY :UINT64 :INT64 :FLOAT64."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-kv-type (gguf-context-pointer gguf) key-id)))

;;; KV metadata — value access

(defun gguf-val (gguf key-id)
  "Return the CL value for scalar KV entry KEY-ID in GGUF.
Dispatches on gguf-kv-type: numeric types → integers or floats,
:BOOL → T/NIL, :STRING → string.
Returns :ARRAY for array-typed entries; use GGUF-ARR-* for those."
  (with-llama-compatible-fp-environment
    (let ((ptr (gguf-context-pointer gguf)))
      (ecase (%llama:gguf-get-kv-type ptr key-id)
        (:uint8   (%llama:gguf-get-val-u8   ptr key-id))
        (:int8    (%llama:gguf-get-val-i8   ptr key-id))
        (:uint16  (%llama:gguf-get-val-u16  ptr key-id))
        (:int16   (%llama:gguf-get-val-i16  ptr key-id))
        (:uint32  (%llama:gguf-get-val-u32  ptr key-id))
        (:int32   (%llama:gguf-get-val-i32  ptr key-id))
        (:float32 (%llama:gguf-get-val-f32  ptr key-id))
        (:uint64  (%llama:gguf-get-val-u64  ptr key-id))
        (:int64   (%llama:gguf-get-val-i64  ptr key-id))
        (:float64 (%llama:gguf-get-val-f64  ptr key-id))
        (:bool    (not (zerop (%llama:gguf-get-val-bool ptr key-id))))
        (:string  (%llama:gguf-get-val-str  ptr key-id))
        (:array   :array)
        (:count   :count)))))

;;; KV metadata — array access

(defun gguf-arr-type (gguf key-id)
  "Return the element gguf-type keyword for array KV entry KEY-ID."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-arr-type (gguf-context-pointer gguf) key-id)))

(defun gguf-arr-n (gguf key-id)
  "Return the element count of array KV entry KEY-ID."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-arr-n (gguf-context-pointer gguf) key-id)))

(defun %gguf-arr-cffi-type (arr-type)
  "Map a GGUF element type keyword to a CFFI primitive type for MEM-AREF."
  (ecase arr-type
    (:uint8   :unsigned-char)
    (:int8    :char)
    (:uint16  :unsigned-short)
    (:int16   :short)
    (:uint32  :unsigned-int)
    (:int32   :int)
    (:float32 :float)
    (:uint64  :unsigned-long)
    (:int64   :long)
    (:float64 :double)
    (:bool    :char)))

(defun gguf-arr-data (gguf key-id)
  "Return (values foreign-pointer cffi-element-type count) for numeric array KV KEY-ID.
CFFI-ELEMENT-TYPE is a CFFI primitive keyword (e.g. :FLOAT, :UNSIGNED-INT)
suitable for CFFI:MEM-AREF. Not valid for :STRING arrays; use GGUF-ARR-STR."
  (with-llama-compatible-fp-environment
    (let* ((ptr (gguf-context-pointer gguf))
           (arr-type (%llama:gguf-get-arr-type ptr key-id)))
      (when (eq arr-type :string)
        (error "~S is not valid for :STRING arrays; use ~S" 'gguf-arr-data 'gguf-arr-str))
      (values (%llama:gguf-get-arr-data ptr key-id)
              (%gguf-arr-cffi-type arr-type)
              (%llama:gguf-get-arr-n ptr key-id)))))

(defun gguf-arr-str (gguf key-id index)
  "Return the string at INDEX within :STRING array KV entry KEY-ID."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-arr-str (gguf-context-pointer gguf) key-id index)))

;;; Type name

(defun gguf-type-name (type)
  "Return a human-readable string for the gguf-type keyword TYPE."
  (with-llama-compatible-fp-environment
    (%llama:gguf-type-name type)))

;;; Convenience aggregate

(defun gguf-metadata (gguf)
  "Return all KV metadata from GGUF as an alist of (key . value) pairs.
Scalar values are resolved to CL types. Array entries appear as (key . (:array . N))."
  (with-llama-compatible-fp-environment
    (let* ((ptr (gguf-context-pointer gguf))
           (n (%llama:gguf-get-n-kv ptr)))
      (loop for i from 0 below n
            for key = (%llama:gguf-get-key ptr i)
            for kv-type = (%llama:gguf-get-kv-type ptr i)
            collect (cons key
                          (if (eq kv-type :array)
                              (cons :array (%llama:gguf-get-arr-n ptr i))
                              (ecase kv-type
                                (:uint8   (%llama:gguf-get-val-u8   ptr i))
                                (:int8    (%llama:gguf-get-val-i8   ptr i))
                                (:uint16  (%llama:gguf-get-val-u16  ptr i))
                                (:int16   (%llama:gguf-get-val-i16  ptr i))
                                (:uint32  (%llama:gguf-get-val-u32  ptr i))
                                (:int32   (%llama:gguf-get-val-i32  ptr i))
                                (:float32 (%llama:gguf-get-val-f32  ptr i))
                                (:uint64  (%llama:gguf-get-val-u64  ptr i))
                                (:int64   (%llama:gguf-get-val-i64  ptr i))
                                (:float64 (%llama:gguf-get-val-f64  ptr i))
                                (:bool    (not (zerop (%llama:gguf-get-val-bool ptr i))))
                                (:string  (%llama:gguf-get-val-str  ptr i))
                                (:count   :count))))))))

;;; Tensor info

(defun gguf-n-tensors (gguf)
  "Return the number of tensors described in GGUF."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-n-tensors (gguf-context-pointer gguf))))

(defun gguf-find-tensor (gguf name)
  "Return the integer index of tensor NAME in GGUF, or NIL if not present."
  (with-llama-compatible-fp-environment
    (let ((id (%llama:gguf-find-tensor (gguf-context-pointer gguf) name)))
      (unless (minusp id) id))))

(defun gguf-tensor-name (gguf tensor-id)
  "Return the name string of tensor TENSOR-ID in GGUF."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-tensor-name (gguf-context-pointer gguf) tensor-id)))

(defun gguf-tensor-type (gguf tensor-id)
  "Return the ggml-type keyword for tensor TENSOR-ID (e.g. :F32, :Q4-0)."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-tensor-type (gguf-context-pointer gguf) tensor-id)))

(defun gguf-tensor-offset (gguf tensor-id)
  "Return the byte offset of tensor TENSOR-ID within the data section."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-tensor-offset (gguf-context-pointer gguf) tensor-id)))

(defun gguf-tensor-size (gguf tensor-id)
  "Return the byte size of tensor TENSOR-ID."
  (with-llama-compatible-fp-environment
    (%llama:gguf-get-tensor-size (gguf-context-pointer gguf) tensor-id)))

(defun gguf-tensor-info (gguf tensor-id)
  "Return a plist of properties for tensor TENSOR-ID.
Keys: :NAME (string), :TYPE (ggml-type keyword), :OFFSET (bytes), :SIZE (bytes)."
  (with-llama-compatible-fp-environment
    (let ((ptr (gguf-context-pointer gguf)))
      (list :name   (%llama:gguf-get-tensor-name   ptr tensor-id)
            :type   (%llama:gguf-get-tensor-type   ptr tensor-id)
            :offset (%llama:gguf-get-tensor-offset ptr tensor-id)
            :size   (%llama:gguf-get-tensor-size   ptr tensor-id)))))

(defun gguf-tensors (gguf)
  "Return a list of plists, one per tensor in GGUF.
Each plist has :NAME, :TYPE, :OFFSET, and :SIZE keys."
  (with-llama-compatible-fp-environment
    (let* ((ptr (gguf-context-pointer gguf))
           (n (%llama:gguf-get-n-tensors ptr)))
      (loop for i from 0 below n
            collect (list :name   (%llama:gguf-get-tensor-name   ptr i)
                          :type   (%llama:gguf-get-tensor-type   ptr i)
                          :offset (%llama:gguf-get-tensor-offset ptr i)
                          :size   (%llama:gguf-get-tensor-size   ptr i))))))
