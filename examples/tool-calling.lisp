;;;; tool-calling.lisp
;;;;
;;;; Demonstrates how to implement tool/function calling with cl-llama-cpp.
;;;; No new library dependencies are required — JSON schemas are built with
;;;; FORMAT and tool-call output is parsed with simple string operations.
;;;; In production code, use a JSON library (jzon, yason, cl-json) instead.
;;;;
;;;; Workflow
;;;; --------
;;;;  1. Define tool schemas as a JSON array (OpenAI-compatible format)
;;;;  2. Embed the schemas in the system message — the chat template formats
;;;;     them correctly for each model family
;;;;  3. Generate — the model emits a tool call in its native JSON format
;;;;  4. Parse the tool name and arguments out of the generated text
;;;;  5. Call the corresponding Lisp function and capture the result
;;;;  6. Append both the assistant turn (the tool call) and a "tool" message
;;;;     (the result) to the conversation, then generate again
;;;;  7. The model's second response is its final answer
;;;;
;;;; Tool call output format
;;;; -----------------------
;;;; Different model families emit tool calls differently.  This example
;;;; instructs the model via the system prompt to use a specific format:
;;;;
;;;;   <tool_call>{"name": "...", "arguments": {...}}</tool_call>
;;;;
;;;; That makes parsing deterministic regardless of the model's native
;;;; template.  For models with native tool-call support (e.g. Llama-3.1,
;;;; Qwen-2.5) you can instead rely on the chat template and parse its
;;;; output directly — see the "Native chat-template tool calling" note below.
;;;;
;;;; Run from a shell:
;;;;   ros -e '(ql:quickload :cl-llama-cpp/examples)' \
;;;;       -e '(cl-llama-cpp/examples/tool-calling:main :model-path "/path/to/model.gguf")' -q

(defpackage #:cl-llama-cpp/examples/tool-calling
  (:use #:cl #:cl-llama-cpp)
  (:export #:main #:run-demo))

(in-package #:cl-llama-cpp/examples/tool-calling)

;;; ---------------------------------------------------------------------------
;;; Tool implementations — ordinary Lisp functions
;;; ---------------------------------------------------------------------------

(defun get-weather (location)
  "Simulate a weather API call."
  (format nil "72°F and sunny in ~A" location))

(defun get-time (timezone)
  "Simulate a time API call."
  (format nil "15:42 ~A" timezone))

;;; ---------------------------------------------------------------------------
;;; Tool schemas
;;;
;;; Build a JSON array of tool definitions in the OpenAI function-calling
;;; schema.  FORMAT works fine for small, fixed schemas.  For dynamic schemas
;;; or many tools, use a JSON library (jzon: (jzon:stringify schema)).
;;; ---------------------------------------------------------------------------

(defparameter *tool-schemas*
  (format nil
    "[~%~
      {\"type\":\"function\",\"function\":{~%~
        \"name\":\"get_weather\",~%~
        \"description\":\"Get the current weather for a location.\",~%~
        \"parameters\":{\"type\":\"object\",~%~
          \"properties\":{~%~
            \"location\":{\"type\":\"string\",\"description\":\"City and country, e.g. Paris, France\"}~%~
          },~%~
          \"required\":[\"location\"]}~%~
      }},~%~
      {\"type\":\"function\",\"function\":{~%~
        \"name\":\"get_time\",~%~
        \"description\":\"Get the current time in a timezone.\",~%~
        \"parameters\":{\"type\":\"object\",~%~
          \"properties\":{~%~
            \"timezone\":{\"type\":\"string\",\"description\":\"Timezone name, e.g. UTC, America/New_York\"}~%~
          },~%~
          \"required\":[\"timezone\"]}~%~
      }}~%~
    ]"))

;;; ---------------------------------------------------------------------------
;;; System message
;;;
;;; Embedding the tool schemas in the system message works for any
;;; instruction-tuned model and lets you control the exact output format.
;;; ---------------------------------------------------------------------------

(defun make-system-prompt (tools-json)
  (format nil
    "You are a helpful assistant with access to the following tools:~%~%~
     ~A~%~%~
     When you need to use a tool, respond with ONLY a single XML tag in ~
     this exact format (no other text before or after):~%~%~
     <tool_call>{\"name\": \"<function-name>\", \"arguments\": {<arguments-as-json>}}</tool_call>~%~%~
     After receiving the tool result, provide a helpful, conversational ~
     final answer to the user."
    tools-json))

;;; ---------------------------------------------------------------------------
;;; Tool-call parsing
;;;
;;; Looks for <tool_call>...</tool_call> in the model's output and extracts
;;; the function name and arguments JSON object.
;;; ---------------------------------------------------------------------------

(defun find-between (text open close &key (start 0))
  "Return the substring between the first OPEN marker and the following CLOSE,
or NIL if either is absent."
  (let* ((open-pos (search open text :start2 start))
         (content-start (and open-pos (+ open-pos (length open))))
         (close-pos (and content-start (search close text :start2 content-start))))
    (and close-pos (subseq text content-start close-pos))))

(defun extract-json-object (text start)
  "Return the JSON object in TEXT beginning at brace position START.
Handles nested objects and strings (including escaped quotes)."
  (let ((depth 0) (in-string nil) (escape nil))
    (loop for i from start below (length text)
          for ch = (char text i)
          do (cond
               (escape       (setf escape nil))
               (in-string    (cond ((char= ch #\\) (setf escape t))
                                   ((char= ch #\") (setf in-string nil))))
               ((char= ch #\") (setf in-string t))
               ((char= ch #\{) (incf depth))
               ((char= ch #\}) (decf depth)
                (when (zerop depth)
                  (return (subseq text start (1+ i))))))
          finally (return nil))))

(defun json-string-value (json key)
  "Extract the string value for KEY from a flat JSON object.
Returns NIL if the key is absent or its value is not a quoted string."
  (let* ((search-str (concatenate 'string "\"" key "\""))
         (key-pos (search search-str json)))
    (when key-pos
      (let* ((after-key (+ key-pos (length search-str)))
             (colon-pos (position #\: json :start after-key))
             (q-open    (and colon-pos
                             (position #\" json :start (1+ colon-pos)))))
        (when q-open
          (let ((q-close (position #\" json :start (1+ q-open))))
            (and q-close (subseq json (1+ q-open) q-close))))))))

(defun parse-one-tool-call (payload)
  "Parse a single tool-call JSON payload string. Returns (name . args-json) or NIL."
  (let ((name (json-string-value payload "name"))
        (args-start (search "\"arguments\"" payload)))
    (when (and name args-start)
      (let* ((brace-pos (position #\{ payload :start args-start))
             (args-json (and brace-pos (extract-json-object payload brace-pos))))
        (when args-json
          (cons name args-json))))))

(defun parse-all-tool-calls (text)
  "Return a list of (name . arguments-json) for every <tool_call>…</tool_call>
block in TEXT, in order."
  (let ((pos 0) (results '()))
    (loop
      (let* ((open (search "<tool_call>" text :start2 pos))
             (content-start (and open (+ open (length "<tool_call>"))))
             (close (and content-start (search "</tool_call>" text :start2 content-start))))
        (unless (and open close)
          (return))
        (let ((call (parse-one-tool-call (subseq text content-start close))))
          (when call (push call results)))
        (setf pos (+ close (length "</tool_call>")))))
    (nreverse results)))

;;; ---------------------------------------------------------------------------
;;; Tool dispatch
;;; ---------------------------------------------------------------------------

(defun call-tool (name arguments-json)
  "Dispatch to the Lisp function named NAME with args drawn from ARGUMENTS-JSON.
Returns a result string, or an error description if the tool is unknown."
  (cond
    ((string= name "get_weather")
     (get-weather (or (json-string-value arguments-json "location") "unknown")))
    ((string= name "get_time")
     (get-time (or (json-string-value arguments-json "timezone") "UTC")))
    (t
     (format nil "Error: unknown tool ~S" name))))

;;; ---------------------------------------------------------------------------
;;; Tool-result message
;;;
;;; Append both the assistant turn (the raw tool-call text) and a "tool"
;;; message containing the result before the second generate call.
;;;
;;; Note: modern instruction-tuned models (Llama-3.1, Qwen-2.5, Mistral-v3)
;;; understand the "tool" role directly.  If your model does not, wrap the
;;; result in a "user" message instead.
;;; ---------------------------------------------------------------------------

(defun tool-result-message (tool-name result)
  (list :role "tool"
        :content (format nil "Result of ~A: ~A" tool-name result)))

;;; ---------------------------------------------------------------------------
;;; Demo
;;; ---------------------------------------------------------------------------

(defun run-demo (&key (model-path "/path/to/model.gguf")
                      (n-gpu-layers 99)
                      (n-ctx 4096)
                      (max-tokens 512)
                      (max-tool-turns 5)
                      (user-query "What's the weather in Tokyo and what time is it in UTC?"))
  "Run a tool-calling loop until the model's response contains no tool calls.
Each turn: generate → parse all <tool_call> blocks → dispatch each → append
results → repeat.  MAX-TOOL-TURNS caps the loop to prevent infinite cycles."
  (format t "Query: ~A~%~%" user-query)
  (finish-output)

  (with-backend ()
    (set-log-callback (lambda (level text)
                        (when (>= level 3)
                          (format *error-output* "~a" text))))
    (with-model (model model-path :n-gpu-layers n-gpu-layers)
      (with-context (ctx model :n-ctx n-ctx)
        (let ((messages (list (list :role "system"
                                    :content (make-system-prompt *tool-schemas*))
                              (list :role "user"
                                    :content user-query))))
          (loop for turn from 1
                for response = (generate ctx nil
                                         :prompt-tokens (tokenize-chat model messages)
                                         :max-tokens max-tokens
                                         :temp 0.0)
                do
                  (format t "=== Turn ~D: model response ===~%" turn)
                  (format t "~A~%~%" response)
                  (finish-output)
                  (let ((calls (parse-all-tool-calls response)))
                    (when (or (null calls) (> turn max-tool-turns))
                      (when (> turn max-tool-turns)
                        (format t "[Max tool turns (~D) reached]~%~%" max-tool-turns))
                      (return response))
                    ;; Append this assistant turn, then dispatch every tool call
                    ;; and append each result before the next generate.
                    (setf messages
                          (append messages
                                  (list (list :role "assistant" :content response))))
                    (dolist (call calls)
                      (let* ((name   (car call))
                             (args   (cdr call))
                             (result (call-tool name args)))
                        (format t "=== Tool: ~A ~A ===~%" name args)
                        (format t "=== Result: ~A ===~%~%" result)
                        (finish-output)
                        (setf messages
                              (append messages
                                      (list (tool-result-message name result)))))))))))))


(defun main (&key (model-path "/path/to/model.gguf")
                  (n-gpu-layers 99)
                  (n-ctx 4096)
                  (max-tokens 512)
                  (max-tool-turns 5)
                  user-query)
  "Entry point for shell invocation.

Example:
  ros -e '(ql:quickload :cl-llama-cpp/examples)' \\
      -e '(cl-llama-cpp/examples/tool-calling:main :model-path \"/path/to/model.gguf\")' -q"
  (apply #'run-demo
         :model-path model-path
         :n-gpu-layers n-gpu-layers
         :n-ctx n-ctx
         :max-tokens max-tokens
         :max-tool-turns max-tool-turns
         (when user-query (list :user-query user-query))))

;;;; ---------------------------------------------------------------------------
;;;; Note: Native chat-template tool calling
;;;;
;;;; Some models (Llama-3.1, Qwen-2.5, Hermes-3, …) have tool-calling built
;;;; into their chat template.  With those models you can pass the tool schemas
;;;; as JSON directly inside a "tools" field of the system message using the
;;;; model's expected schema rather than the generic system-prompt approach
;;;; above — the template then formats the tool call markers itself.
;;;;
;;;; The parsing side still requires handling the model's native output format
;;;; (e.g. Llama-3.1 emits {"name": …, "parameters": …} without a wrapper
;;;; tag; Qwen wraps in <tool_call>…</tool_call>).  Inspect a few sample
;;;; outputs from your target model and adjust PARSE-TOOL-CALL accordingly.
;;;;
;;;; The TOKENIZE-CHAT + GENERATE pattern shown above works unchanged — only
;;;; the system message content and the output parser change per model family.
;;;; ---------------------------------------------------------------------------
