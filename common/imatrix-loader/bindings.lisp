(in-package #:cl-llama-cpp/common/imatrix-loader)

;;; Lifecycle
(cffi:defcfun ("llama_extras_imatrix_create" %imatrix-create) :pointer)
(cffi:defcfun ("llama_extras_imatrix_free" %imatrix-free) :void
  (ptr :pointer))
(cffi:defcfun ("llama_extras_imatrix_load" %imatrix-load) :int
  (ptr :pointer) (fname :string))

;;; Scalar queries
(cffi:defcfun ("llama_extras_imatrix_chunk_count" %imatrix-chunk-count) :int32
  (ptr :pointer))
(cffi:defcfun ("llama_extras_imatrix_chunk_size" %imatrix-chunk-size) :int32
  (ptr :pointer))
(cffi:defcfun ("llama_extras_imatrix_is_legacy" %imatrix-is-legacy) :int
  (ptr :pointer))
(cffi:defcfun ("llama_extras_imatrix_has_metadata" %imatrix-has-metadata) :int
  (ptr :pointer))

;;; Datasets
(cffi:defcfun ("llama_extras_imatrix_n_datasets" %imatrix-n-datasets) :int32
  (ptr :pointer))
(cffi:defcfun ("llama_extras_imatrix_dataset" %imatrix-dataset) :string
  (ptr :pointer) (index :int32))

;;; Entries
(cffi:defcfun ("llama_extras_imatrix_n_entries" %imatrix-n-entries) :int32
  (ptr :pointer))
(cffi:defcfun ("llama_extras_imatrix_entry_name" %imatrix-entry-name) :string
  (ptr :pointer) (index :int32))

;;; Entry data
(cffi:defcfun ("llama_extras_imatrix_entry_n_sums" %imatrix-entry-n-sums) :int32
  (ptr :pointer) (index :int32))
(cffi:defcfun ("llama_extras_imatrix_entry_sums" %imatrix-entry-sums) :pointer
  (ptr :pointer) (index :int32))
(cffi:defcfun ("llama_extras_imatrix_entry_n_counts" %imatrix-entry-n-counts) :int32
  (ptr :pointer) (index :int32))
(cffi:defcfun ("llama_extras_imatrix_entry_counts" %imatrix-entry-counts) :pointer
  (ptr :pointer) (index :int32))
