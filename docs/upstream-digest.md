# llama.cpp Upstream Digest

Pin at last update: 4988f6e866057afd130c1515ecef0c9bab9a15f8
Last covered upstream commit: 9d57ce456c94d241dde672b2db9cf18879766568

---

## 2026-08-15 — 812 commits since pin

### New Features

- **DeepSeek V4 architecture** (#24162): Full support for the DeepSeek V4 model family, including hyper-connections, lightning indexer, and fused ops. This is a large new architecture with custom attention patterns and MoE routing.
- **DeepSeek V4 MTP + DSpark speculative decoding** (#25784, #25173): Adds speculative decoding via "DSpark" sidecar draft models for DeepSeek V4, plus MTP (multi-token prediction) layers.
- **DFlash speculative decoding** (#22105): New speculative decoding method using "DFlash" draft models, supported across several architectures including Nemotron 3.5 (#26905).
- **Eagle3 speculative decoding** for Qwen3.5/3.6 (#24593), GPT-OSS (#25794), Minimax2, and others. A tree-based draft approach.
- **MiniMax-M3 (MSA)** (#24908): MiniMax Sparse Attention architecture — a new sparse attention mechanism with dedicated memory implementation (#26338).
- **MiniMaxText01 / MiniMaxM1** (#27018): Recurrent + attention hybrid model with logits masking.
- **GLM 5.2 with DSA Indexer** (#25407, #25980): GLM model with a learned indexer for dynamic sparse attention, plus vision support (#26126).
- **Granite-Switch** (#25107): IBM's mixture-of-experts switch architecture.
- **Granite Speech Plus** (#24818): Speech-capable Granite model.
- **Hy3 (hy_v3) with MTP** (#25395): Hunyuan V3 model with multi-token prediction speculative decoding.
- **Qwen3-TTS** (#26254): Text-to-speech model for Qwen3 family (note: breaking change to llama-tts binary).
- **Qwen3-Next MTP** (#25589), **DeepSeek V3.2 MTP** (#26457), **GLM-4.7-Flash MTP** (#24868), **Step3.5/3.7 Flash MTP** (#24340), **Nemotron MTP** (#26725), **MiMo V2 MTP** (#26412): Multi-token prediction layers added to many existing model architectures.
- **Muse Glimmer** (#26841): New model architecture with tool-calling support.
- **Laguna XS.2, M.1, S-2.1** (#25165, #26232): New Laguna model family.
- **Nanbeige4.2** (#25994), **LFM2.5 variants** (#24913, #25008), **pocket-tts** (#26871): Additional new model support.
- **Q2_0 quantization** (#24448): New 2-bit quantization type with CPU, CUDA (#25707), Metal (#25419), Vulkan (#25430), and OpenCL (#25160) backends.
- **Multi-output backend sampling** (#25532): Allows samplers to run on GPU for multi-output decoding. The penalties sampler got a CUDA backend (#25262).
- **Load mode refactoring** (#20834, #26081): The old `use_mmap`/`use_mlock`/`use_direct_io` booleans in `llama_model_params` are replaced by a single `llama_load_mode` enum with values AUTO, NONE, MMAP, MLOCK, MMAP_MLOCK, DIRECT_IO. AUTO avoids mmap on integrated GPUs.
- **`llama_version()` API**: Returns the library version string at runtime.
- **`llama_ftype_name()` / `llama_model_ftype()`** (#25134): Query and display a model's quantization type as a human-readable string.
- **`llama_vocab_get_suppress_tokens()`**: Access model-specific suppress tokens from GGUF metadata.
- **`llama_sampler_copy()`**: Copy mutable sampler state between two sampler instances of the same type, without cloning.
- **`llama_model_n_layer_nextn()`**: Query the number of MTP (next-N) layers in a model.
- **Semantic versioning** (#26839): llama.cpp now uses semantic versioning via CMake.
- **Server: MCP stdio support** (#26062): The server can now act as an MCP (Model Context Protocol) tool provider.
- **Server: tool isolation via Docker/Podman/SSH** (#26507, #26774): Run tool calls in sandboxed containers.
- **Server: LRU scheduler** (#26572): Multi-model router with LRU-based model eviction.
- **Server: model management API** (#23976): Download, load, and unload models via REST API.
- **Server: /responses API streaming** (#25348): Timings and progress in the /responses endpoint.
- **Server: real-time model load progress** (#24828): SSE feed for model loading status.
- **CLI: HTTP-based implementation** (#24948): The CLI tool now connects to a local server over HTTP instead of linking the library.
- **`llama` app with download subcommand** (#24982): A unified `llama` binary with model downloading.
- **`n_outputs_max_per_seq` context param**: New field in `llama_context_params` to limit outputs per sequence.

### Bug Fixes

- **Sampler `n_vocab` parameter added to penalties** (#26520): `llama_sampler_init_penalties` now takes `n_vocab` as its first argument, decoupling it from the sampler data struct. The old `-1 = context size` option for `penalty_last_n` is removed.
- **DRY sampler drops `n_ctx_train`** (#26524): `llama_sampler_init_dry` no longer takes `n_ctx_train`; the "full-context windows" concept is removed from history-based samplers.
- **SWA not enabled for EXAONE 4.5** (#26848): Sliding window attention was incorrectly disabled.
- **KV cache quantization fix for DSV4** (#25202): Quantized KV caches could produce incorrect results.
- **Flash attention precision on Gemma E4B MTP** (#25148): Wrong precision settings caused bad outputs.
- **Speculative decoding crash on long prompts for Eagle3** (#24707): Out-of-bounds access fixed.
- **OOB reads in UGM tokenizer** (#18750): Out-of-bounds reads in precompiled_charsmap handling.
- **Seq-rm fix for DeepSeek V4** (#25588): Cache sequence removal was incorrect.
- **Metal NORM/RMS_NORM for partial simdgroups** (#26708): Incorrect results for certain row lengths.
- **CPU affinity mask ignored on Android** (#26838): Thread affinity settings were not being applied.
- **CUDA thread/block count in quantized copy kernels** (#26731): Incorrect launch parameters.
- **Integer overflows in binary ops CUDA** (#24706): Large tensors could trigger truncation.
- **CUDA data races in block_reduce** (#26385): Shared memory reuse race condition.
- **Grammar: degrade max repetition >= 2000 to unbounded** (#26613): Prevents pathological grammar performance.
- **Various SYCL, Vulkan, OpenCL, Metal, and HIP backend fixes** across dozens of commits.

### Capability Gaps

These upstream changes touch the llama.cpp public API but are **not present** in `src/bindings.lisp` or `*binding-deps*`:

- **BREAKING — `llama_sampler_init_penalties` signature changed**: Now requires `n_vocab` as its first argument: `(n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present)`. The existing CFFI binding has the old 4-argument signature and **will crash or produce wrong results** when called against a new build.
- **BREAKING — `llama_sampler_init_dry` signature changed**: The `n_ctx_train` parameter was removed. Old signature: `(vocab, n_ctx_train, dry_multiplier, ...)`. New: `(vocab, dry_multiplier, ...)`. The existing binding passes an extra argument.
- **BREAKING — `llama_model_params` struct layout changed**: `use_mmap`, `use_direct_io`, `use_mlock` (three bools) replaced by `load_mode` (enum `llama_load_mode`). A `load_mtp` bool was added. The struct size and field offsets have shifted.
- **BREAKING — `llama_context_params` struct layout changed**: `n_outputs_max_per_seq` (uint32_t) was inserted after `n_outputs_max`, shifting all subsequent fields.
- **New function `llama_version()`**: Returns a version string. Not bound.
- **New function `llama_ftype_name()`**: Converts `llama_ftype` enum to human-readable string (e.g. "Q8_0"). Not bound.
- **New function `llama_model_ftype()`**: Returns the model's quantization type as enum. Not bound.
- **New functions `llama_load_mode_name()` / `llama_load_mode_from_str()`**: Convert between the new `llama_load_mode` enum and strings. Not bound.
- **New function `llama_model_n_layer_nextn()`**: Returns the number of MTP (next-N prediction) layers. Not bound.
- **New function `llama_vocab_get_suppress_tokens()`**: Returns model-specific suppress tokens from GGUF metadata. Not bound.
- **New function `llama_sampler_copy()`**: Copies mutable sampler state from src to dst without cloning. Not bound.
- **New enum `llama_load_mode`**: AUTO/NONE/MMAP/MLOCK/MMAP_MLOCK/DIRECT_IO. Not defined in bindings.
- **New enum value `LLAMA_FTYPE_MOSTLY_Q2_0` (= 41)**: For the new Q2_0 quantization. Not in enum definition.
- **New op `GGML_OP_LIGHTNING_INDEXER`** (#24231): Implements DSV4 lightning indexer. Not in ggml-op enum if bindings track ops.

### Other / Internal

- **Build**: Semantic versioning via CMake (#26839); GGML versions bumped to 0.15.2 through 0.20.0; BoringSSL updated multiple times; cpp-httplib updated to 0.48–0.53; subprocess.h vendored and patched.
- **CI**: ROCm 7.14 target (#25775); Ubuntu-ROCm disabled (#26969); Windows ROCm in check-release (#26897); SYCL in check-release (#24583); thread sanitizer fixes; ccache removed.
- **CUDA**: CUDA graphs on Volta+Turing (#25749); virtual device support (#25228); fused RMS_NORM+MUL+ROPE (#26767); WKV7 warp-per-row kernel (#26111); various MMQ refactors.
- **Metal**: Q2_0/TQ2_0 support; BF16 repeat/concat; depthwise conv2d; rope_back; FWHT kernel; lightning indexer; snake activation fusion; col2im_1d.
- **Vulkan**: Q2_0 support; gated_delta_net; POOL_1D; col2im_1d; CONV_3D; GET_ROWS_BACK; numerous MoE and FA optimizations.
- **SYCL**: Flash attention via oneDNN; fused GLU/RMS_NORM+MUL; Q2 mul_mat; conv_2d/3d; tensor split mode; MoE reorder support; various type support expansions.
- **OpenCL (Adreno)**: Extensive Adreno GPU optimization: MoE prefill GEMM, FA decode, dp4a dense kernels, compiled kernel binary caching, noshuffle format fixes.
- **Hexagon**: MUL_MAT tiling rework, flash attention rewrite, L2 cache dirty-bit tracking, CLAMP/VISION_ROPE ops, op-trace support.
- **WebGPU**: FA improvements for quantized KV; NVFP4 support; conv2d_dw kernel; MTP inference optimization.
- **Server UI**: Major UI refactoring (stores, constants, contexts, styles); agentic features (filesystem mentions, CWD, slash commands); MCP servers settings; conversation import/export; model load progress bar; symbolic math sandbox; mobile improvements.
- **Converters**: Many converter fixes for new models; split MTP export; endianness conversion for Q1/TQ2; per_layer_config handling.
- **Docs**: AI-generated code policy updated; conda-forge instructions; maintainer PR link.
- **Refactoring**: Fused ops refactor (#24646); model loading refactor (#24980); prompt cache state ownership (#25649); batch construction refactor (#24843).
