(in-package #:cl-llama-cpp/common/imatrix-loader)

;;; Conditions

(define-condition imatrix-load-error (cl-llama-cpp:llama-error)
  ((filename :initarg :filename :reader imatrix-load-error-filename))
  (:report (lambda (c s)
             (format s "Failed to load importance matrix from ~A"
                     (imatrix-load-error-filename c)))))

;;; Data structures

(defstruct (imatrix-entry (:constructor %make-imatrix-entry))
  (sums #() :type simple-array :read-only t)
  (counts #() :type simple-array :read-only t))

(defstruct (imatrix (:constructor %make-imatrix))
  (entries (make-hash-table :test 'equal) :type hash-table :read-only t)
  (datasets nil :type list :read-only t)
  (chunk-count 0 :type (signed-byte 32) :read-only t)
  (chunk-size 0 :type (signed-byte 32) :read-only t)
  (legacy-p nil :type boolean :read-only t)
  (has-metadata-p nil :type boolean :read-only t))

;;; Unmarshaling helpers

(defun %read-sums (ptr index)
  (let* ((n (%imatrix-entry-n-sums ptr index))
         (data (%imatrix-entry-sums ptr index))
         (arr (make-array n :element-type 'single-float)))
    (dotimes (i n arr)
      (setf (aref arr i) (cffi:mem-aref data :float i)))))

(defun %read-counts (ptr index)
  (let* ((n (%imatrix-entry-n-counts ptr index))
         (data (%imatrix-entry-counts ptr index))
         (arr (make-array n :element-type '(signed-byte 64))))
    (dotimes (i n arr)
      (setf (aref arr i) (cffi:mem-aref data :int64 i)))))

(defun %unmarshal-imatrix (ptr)
  (let ((entries (make-hash-table :test 'equal))
        (datasets nil))
    (dotimes (i (%imatrix-n-entries ptr))
      (let ((name (%imatrix-entry-name ptr i)))
        (setf (gethash name entries)
              (%make-imatrix-entry :sums (%read-sums ptr i)
                                   :counts (%read-counts ptr i)))))
    (dotimes (i (%imatrix-n-datasets ptr))
      (push (%imatrix-dataset ptr i) datasets))
    (%make-imatrix
     :entries entries
     :datasets (nreverse datasets)
     :chunk-count (%imatrix-chunk-count ptr)
     :chunk-size (%imatrix-chunk-size ptr)
     :legacy-p (not (zerop (%imatrix-is-legacy ptr)))
     :has-metadata-p (not (zerop (%imatrix-has-metadata ptr))))))

;;; Public API

(defun load-imatrix (filename)
  "Load an importance matrix from FILENAME (a .dat or .gguf file).
Returns an IMATRIX struct containing all entries, datasets, and metadata.
Signals IMATRIX-LOAD-ERROR if the file cannot be loaded."
  (let* ((path (merge-pathnames filename))
         (fname (namestring path))
         (ptr (%imatrix-create)))
    (when (cffi:null-pointer-p ptr)
      (error 'imatrix-load-error :filename fname))
    (unwind-protect
         (progn
           (when (zerop (%imatrix-load ptr fname))
             (error 'imatrix-load-error :filename fname))
           (%unmarshal-imatrix ptr))
      (%imatrix-free ptr))))
