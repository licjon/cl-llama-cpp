(in-package #:cl-llama-cpp/common/shim)

(defun %build-shim-if-needed ()
  "Build the unified common shim .so if it is missing or older than
libllama-common.so or any shim source file."
  (let* ((root-dir   (asdf:system-source-directory "cl-llama-cpp"))
         (build-dir  (merge-pathnames "llama.cpp/build/bin/" root-dir))
         (shim-so    (merge-pathnames "libllama-common-shim.so" build-dir))
         (common-so  (merge-pathnames "libllama-common.so" build-dir))
         (shim-srcs  (directory (merge-pathnames "shim/*.cpp" root-dir)))
         (old-shims  (list (merge-pathnames "libllama-extras-shim.so" build-dir)
                            (merge-pathnames "libllama-json-schema-shim.so" build-dir))))
    (when (and (probe-file common-so)
               shim-srcs
               (or (not (probe-file shim-so))
                   (> (file-write-date common-so) (file-write-date shim-so))
                   (some (lambda (src)
                           (> (file-write-date src) (file-write-date shim-so)))
                         shim-srcs)))
      (format *error-output* "~&; Building libllama-common-shim.so...~%")
      (let ((makefile-dir (merge-pathnames "shim/" root-dir)))
        (multiple-value-bind (output error-output exit-code)
            (uiop:run-program
             (list "make" "-C" (namestring makefile-dir)
                   (format nil "LLAMA_CPP_DIR=~A"
                           (namestring (merge-pathnames "llama.cpp/" root-dir))))
             :output :string :error-output :string :ignore-error-status t)
          (declare (ignore output))
          (unless (zerop exit-code)
            (error "Failed to build common shim (exit ~D):~%~A"
                   exit-code error-output))
          (format *error-output* "; Done.~%")))
      (dolist (old-shim old-shims)
        (when (probe-file old-shim)
          (handler-case (delete-file old-shim)
            (error (c)
              (format *error-output*
                      "; Warning: could not remove old shim ~A: ~A~%"
                      old-shim c))))))))

(%build-shim-if-needed)

(cffi:define-foreign-library libllama-common-shim
  (:unix (:or "libllama-common-shim.so"))
  (:darwin (:or "libllama-common-shim.dylib"))
  (t (:default "libllama-common-shim")))

(cffi:use-foreign-library libllama-common-shim)
