(defpackage #:cl-llama-cpp/generate
  (:use #:cl)
  (:export #:generate))

(in-package #:cl-llama-cpp/generate)

(defun project-path (relative)
  (namestring (merge-pathnames relative
                               (asdf:system-source-directory "cl-llama-cpp"))))

(defun build-wrapper-form ()
  `(claw:defwrapper (:cl-llama-cpp
                     (:headers "llama.h")
                     (:includes ,(project-path "llama.cpp/include/")
                                ,(project-path "llama.cpp/ggml/include/"))
                     (:spec-path ,(project-path "spec/"))
                     (:include-definitions "^(llama|LLAMA)_\\w+")
                     (:language :c))
     :in-package :%llama
     :trim-enum-prefix t
     :recognize-bitfields t
     :recognize-strings t
     :recognize-arrays t
     :symbolicate-names (:in-pipeline
                         (:by-removing-prefixes "llama_" "LLAMA_"))))

(defun claw-anonymous-p (sym)
  "Return T if SYM lives in the %CLAW.ANONYMOUS package."
  (and (symbolp sym)
       (symbol-package sym)
       (string= (package-name (symbol-package sym)) "%CLAW.ANONYMOUS")))

(defun numeric-id-p (sym)
  "Return T if SYM's name is a bare integer (CLAW anonymous union/struct id)."
  (and (symbolp sym)
       (every #'digit-char-p (symbol-name sym))))

(defun fixup-form (form parent-struct-name)
  "Walk FORM and replace %CLAW.ANONYMOUS symbols and numeric-ID type names
with descriptive names derived from PARENT-STRUCT-NAME."
  (cond
    ;; defcunion/defcstruct with numeric ID name -> rename
    ((and (listp form)
          (member (car form) '(cffi:defcunion cffi:defcstruct))
          (let ((name (if (listp (cadr form)) (caadr form) (cadr form))))
            (numeric-id-p name)))
     (let* ((spec (cadr form))
            (old-name (if (listp spec) (car spec) spec))
            (new-name (intern (format nil "~A-VALUE"
                                      (string-upcase
                                       (or parent-struct-name "ANONYMOUS")))
                              (find-package :%llama))))
       ;; Return: (values new-form old-name new-name)
       (values
        (cons (car form)
              (cons (if (listp spec)
                        (cons new-name (cdr spec))
                        new-name)
                    (cddr form)))
        old-name
        new-name)))
    (t (values form nil nil))))

(defun find-users-of-numeric-ids (forms)
  "Scan FORMS to find which named struct/union references each numeric-ID type.
Returns an alist of (numeric-id-symbol . parent-struct-name-string)."
  (let ((result nil))
    (dolist (form forms)
      (when (and (listp form)
                 (member (car form) '(cffi:defcstruct cffi:defcunion)))
        (let ((name (if (listp (cadr form)) (caadr form) (cadr form))))
          (unless (numeric-id-p name)
            ;; Scan field types for numeric-ID references
            (labels ((scan (tree)
                       (cond
                         ((and (symbolp tree) (numeric-id-p tree))
                          (unless (assoc tree result)
                            (push (cons tree (symbol-name name)) result)))
                         ((listp tree)
                          (dolist (x tree) (scan x))))))
              (dolist (field-spec (cddr form))
                (scan field-spec)))))))
    result))

(defun sret-result-p (form)
  "Return T if FORM is a defcfun with CLAW's SRET pattern: return type (:struct T)
and a fabricated (result (:struct T)) parameter with the same struct type."
  (and (listp form)
       (eq (car form) 'cffi:defcfun)
       (listp (caddr form))
       (eq (car (caddr form)) :struct)
       ;; Find the parameter list (skip name-spec, return-type, docstring)
       (let* ((after-rettype (cdddr form))
              (params (if (stringp (car after-rettype))
                          (cdr after-rettype)
                          after-rettype))
              (first-param (car params)))
         (and first-param
              (listp first-param)
              (eq (car first-param) 'result)
              (equal (cadr first-param) (caddr form))))))

(defun strip-sret-result (form)
  "Remove the fabricated (result (:struct T)) parameter from an SRET defcfun.
Returns the corrected form."
  (let* ((name-spec (cadr form))
         (return-type (caddr form))
         (after-rettype (cdddr form))
         (docstring (when (stringp (car after-rettype))
                      (car after-rettype)))
         (params (if docstring
                     (cdr after-rettype)
                     after-rettype))
         ;; Remove the result param (always first)
         (real-params (cdr params)))
    (if docstring
        `(cffi:defcfun ,name-spec ,return-type ,docstring ,@real-params)
        `(cffi:defcfun ,name-spec ,return-type ,@real-params))))

(defun fixup-expansion (forms)
  "Post-process the list of CLAW-generated forms to eliminate %CLAW.ANONYMOUS
references and numeric-ID union/struct names."
  (let ((renames nil)   ; alist of (old-name . new-name)
        (result nil)
        ;; Build map: numeric-id -> parent-struct-name by scanning who references them
        (id-to-parent (find-users-of-numeric-ids forms)))
    ;; Process forms
    (dolist (form forms)
      (cond
        ;; defcfun with SRET pattern -> strip fabricated result parameter
        ((sret-result-p form)
         (push (strip-sret-result form) result))
        ;; defcunion/defcstruct with numeric ID -> rename based on parent
        ((and (listp form)
              (member (car form) '(cffi:defcunion cffi:defcstruct))
              (let ((name (if (listp (cadr form)) (caadr form) (cadr form))))
                (numeric-id-p name)))
         (let* ((name (if (listp (cadr form)) (caadr form) (cadr form)))
                (parent (cdr (assoc name id-to-parent))))
           (multiple-value-bind (new-form old-name new-name)
               (fixup-form form parent)
             (push (cons old-name new-name) renames)
             (push new-form result))))
        ;; Export of %CLAW.ANONYMOUS symbol -> skip
        ((and (listp form)
              (eq (car form) 'common-lisp:export)
              (listp (cdr form))
              (let ((sym-form (cadr form)))
                (and (listp sym-form)
                     (eq (car sym-form) 'quote)
                     (claw-anonymous-p (cadr sym-form)))))
         nil)
        ;; Export of numeric-ID symbol -> use renamed name
        ((and (listp form)
              (eq (car form) 'common-lisp:export)
              (listp (cdr form))
              (let ((sym-form (cadr form)))
                (and (listp sym-form)
                     (eq (car sym-form) 'quote)
                     (numeric-id-p (cadr sym-form)))))
         (let* ((old-name (cadr (cadr form)))
                (entry (assoc old-name renames)))
           (if entry
               (push `(common-lisp:export ',(cdr entry) "%LLAMA") result)
               (push form result))))
        ;; Any other form -- just keep it
        (t (push form result))))
    ;; Now apply renames in all forms (replace old type refs with new names)
    (setf result (nreverse result))
    (when renames
      (setf result (mapcar (lambda (form)
                             (rename-in-tree form renames))
                           result)))
    result))

(defun rename-in-tree (tree renames)
  "Walk TREE replacing any symbol that appears as a key in RENAMES alist."
  (cond
    ((and (symbolp tree)
          (assoc tree renames))
     (cdr (assoc tree renames)))
    ((and (symbolp tree)
          (claw-anonymous-p tree))
     ;; Replace any remaining %CLAW.ANONYMOUS symbol with 'value
     (intern "VALUE" (find-package :%llama)))
    ((listp tree)
     (mapcar (lambda (x) (rename-in-tree x renames)) tree))
    (t tree)))

(defun write-bindings (expansion output)
  (with-open-file (out output :direction :output
                              :if-exists :supersede
                              :external-format :utf-8)
    (let ((*package* (find-package :%llama))
          (*print-case* :downcase)
          (*print-pretty* t)
          (*print-right-margin* 100))
      (format out ";;;; Auto-generated CFFI bindings for llama.cpp~%")
      (format out ";;;; Generated by cl-llama-cpp/generate -- DO NOT EDIT~%~%")
      (format out "(in-package #:%llama)~%~%")
      ;; Collect all forms first, then fixup, then emit
      (let ((forms nil))
        (labels ((collect (form)
                   (cond
                     ((and (listp form) (eq (car form) 'progn))
                      (dolist (f (cdr form)) (collect f)))
                     (t (push form forms)))))
          (collect expansion))
        (setf forms (nreverse forms))
        (setf forms (fixup-expansion forms))
        (dolist (form forms)
          (pprint form out)
          (terpri out)
          (terpri out))))))

(defun generate (&key (output (project-path "src/bindings.lisp"))
                      rebuild-spec)
  "Generate CFFI bindings from llama.h via CLAW.
When REBUILD-SPEC is true, force c2ffi to re-parse headers even if
spec files exist (use after bumping the llama.cpp submodule)."
  (when rebuild-spec
    (pushnew :claw-rebuild-spec *features*))
  (pushnew :claw-local-only *features*)
  (let* ((form (build-wrapper-form))
         (expansion (macroexpand-1 form)))
    (write-bindings expansion output)
    (format t "~&Bindings written to ~a~%" output)
    output))
