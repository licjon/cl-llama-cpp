# llama.cpp Upstream Digest

Pin at last update: 4988f6e866057afd130c1515ecef0c9bab9a15f8
Last covered upstream commit: 3af988fabcf79fd81f8720505e684d2aa5bfc786

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

---

## 2026-08-21 — 134 commits since last digest

### New Features

- **Kimi-K3 text model** (#26185): Moonshot's Kimi K3 — a hybrid architecture combining KDA (linear/recurrent) attention with MLA (full) attention. Introduces five new mechanisms over Kimi-Linear: cross-layer residual attention, latent MoE (1024 experts, up from the previous max of 512), "situ" activation (replaces SwiGLU), MLA output gating, and a full-rank KDA gate. Includes full chat template with reasoning and tool-call support.
- **BailingMoE3** (#26608): ByteDance's Bailing MoE v3 architecture with speculative decoding support, Q-LoRA (Ling-3.0-tiny), and SwiGLU activation clamping. Includes both flash and tiny model variants.
- **GraniteSWA / GraniteMoeSWA** (#25505): IBM Granite models with sliding window attention and per-layer rope pattern control. Adds a new `has_rope` hparam and `rope_pattern` array for models that mix RoPE and NoPE layers.
- **dots3-note** (#27060): New model architecture with DSA-iSWA (dynamic sparse attention with interleaved sliding window attention) KV cache.
- **DSpark speculators-format support** (#26275): The DSpark speculative decoder now accepts SpecForge-exported drafts with reduced vocabulary, bonus-anchor block layout, and d2t remapping tables.
- **`--mmproj-device` argument** (#23255): Lets you place the vision projector (multimodal) on a specific GPU device instead of the default.
- **`--models-dir` MTP loading** (#24431): The models directory auto-discovery now finds MTP (multi-token prediction) assistant models for speculative decoding.
- **`ggml_rope_set_offset()`** (#27120): New ggml operation that applies RoPE with an explicit position offset, supported on CPU, Metal, CUDA, Vulkan, OpenCL, SYCL, WebGPU, and Hexagon backends.
- **llama.cpp version 0.2.0** (#27498): Major version bump with semantic versioning now enforced via CMake. Includes a `release.sh` script for release preparation.
- **Server: model endpoint auth** (#26347): The `/models` endpoints are now private (require API key) when authentication is enabled.
- **Server: sleep refactor** (#27376): The `/metrics` endpoint remains accessible while the server is sleeping. Sleep state handling cleaned up with cached responses.
- **Server: dedup-cache-models preset** (#27346): New preset option for deduplicating cached models in multi-model setups.
- **LLAMA_MAX_EXPERTS raised to 1024**: Up from 512, needed for Kimi K3's latent MoE.

### Bug Fixes

- **Unicode '~' missing from symbol class** (#26972): The tilde character was not included in the collapsed `\p{S}` (Symbol) Unicode class, so input like `" ~"` was split into separate pre-tokens. This broke the `Ġ~` BPE merge used by DeepSeek V4, causing re-tokenized prompts to diverge from sampled tokens and preventing KV cache reuse.
- **Metal mat-mul OOB read for K not a multiple of 32** (#27450): The Tensor API mat-mat kernel used a static K=32 tile width on every iteration, reading past the K extent on the last partial tile. Could corrupt results or produce NaN.
- **Backend split scheduler race condition** (#26040): Graph splits without inputs were running concurrently with other splits that reused the same memory, causing data corruption.
- **LoRA tensor bounds check** (#27056): LoRA adapter loading now validates that tensor data regions fall within file bounds, preventing OOB reads on malformed adapter files.
- **GGUF array type validation** (#27075): Array element types are now checked before reading, preventing type-confusion bugs on malformed GGUF files.
- **gguf-py size guards** (#27188): The Python GGUF reader now validates kv_count, tensor_count, string length, and array length against crafted values that could cause unbounded allocation or hangs (security fix).
- **Nemotron 3 Ultra block count** (#27101): The converter was miscounting blocks for the Nemotron 3 Ultra architecture.
- **LFM2 image tiling threshold** (#27057): The multimodal preprocessor was using an incorrect threshold for deciding when to tile LFM2 images.
- **SYCL mlock loading** (#27250): Loading models with mlock was broken on SYCL backends.
- **SYCL zero device crash** (#27291): llama-quantize (and other non-compute tools) crashed on hosts with no SYCL devices instead of gracefully falling back.
- **Vulkan null queue cleanup** (#27353): Missing null checks in queue cleanup caused crashes on some Vulkan drivers.
- **Vulkan FA precision for Q types** (#27413): The flash attention MMQ path could overflow when computing `1/qd` for denormalized Q scale values; now uses fp32.
- **OpenCL q6_K on Adreno A6x/A7x** (#26476): Four compiler codegen defects in the q6_K flat mul_mat kernel on older Adreno E031 compilers, worked around with version-gated code paths.
- **OpenCL norm local size** (#27339): Incorrect workgroup size for normalization kernels.
- **OpenCL FA tile kernel race** (#26434): Write-after-read race condition in the generic flash attention tile kernels when the workgroup spans multiple subgroups.
- **Hexagon FA queue ordering** (#27042): Incorrect HMX queue ordering in the pipelined flash attention path, plus D matrix packing fix.
- **Server `--docker-repo` mode detection** (#27416): The `--docker-repo` flag was incorrectly triggering router mode.
- **JSON schema regex fallback** (#26939): Unsupported regex patterns in JSON schema now gracefully degrade instead of failing.
- **Thread pool sharing reverted** (#27337): The "share thread pools when `n_threads` differ" change (#27138) caused issues and was reverted.
- **HIP UMA memory reporting** (#27083): AMD APUs report accurate memory via `hipMemGetInfo`; the UMA override was over-promising available VRAM on small-carveout systems.
- **SYCL quantized copy kernel dimensions** (#27160): Thread/block counts were not proportional to quant size; q4_0→f32 throughput improved from 20 to 158 GB/s on Arc 70.
- **Integer tokenizer scores** (#27260): Tokenizer scores stored as integers in GGUF are now handled correctly.
- **Speculative decoding null reference** (#27404): Binding a reference to a null pointer in the speculative decoding path.

### Capability Gaps

All previously identified breaking API changes (from the 2026-08-15 digest) remain unresolved in `src/bindings.lisp`:

- **`llama_sampler_init_penalties`** still uses the old 4-argument signature (upstream now requires `n_vocab` as the first argument).
- **`llama_sampler_init_dry`** still passes `n_ctx_train` (upstream removed it).
- **`llama_model_params` struct** still has `use_mmap`/`use_direct_io`/`use_mlock` bools (upstream replaced them with `load_mode` enum and added `load_mtp`).
- **`llama_context_params` struct** is missing the `n_outputs_max_per_seq` field.
- **`llama_version()`**, **`llama_ftype_name()`**, **`llama_model_ftype()`**, **`llama_load_mode_name()`/`llama_load_mode_from_str()`**, **`llama_model_n_layer_nextn()`**, **`llama_vocab_get_suppress_tokens()`**, **`llama_sampler_copy()`** — all still unbound.
- **`llama_load_mode` enum** and **`LLAMA_FTYPE_MOSTLY_Q2_0`** — still not defined.

No new llama C API surface changes were introduced in this batch of commits. The migration of `--mmap`/`--no-mmap` CLI flags to `--load-mode` (#26934) is a CLI-level change that reinforces the already-noted struct-level break.

### Other / Internal

- **Build**: llama.cpp version bumped 0.1.1 → 0.1.2 → 0.2.0; ggml bumped 0.20.1 → 0.20.2 → 0.21.0; release workflow overhauled with deploy keys, attestation, configurable commit targeting, and pre-release changelog generation; `release.sh` script added; xcframework builds parallelized and made configurable; BoringSSL updated to 0.20260813.0; cpp-httplib updated to 0.53.1; hash library moved to vendor directory with CMake alias targets.
- **CI**: Windows ARM64 CUDA 13.4 added; OpenVINO updated to 2026.3; release attestation; ccache-clear as last release step; cmake pkg check via shell script; duplicate flags removed; SYCL release dependency re-enabled.
- **CUDA**: Per-hardware MMVQ→MMQ crossover switch points for Blackwell, Ada (RTX 4090), and DGX Spark (#26079); MMVQ nwarps=8 for bs=1 on DGX Spark (#26843); static cuBLAS workspace (#26574).
- **Metal**: Dequantize quantized KV cache (q8_0, q4_0, q4_1, q5_0, q5_1) to F16 before flash attention (#27390) — improves FA on quantized KV by running the proven F16 kernels on a dequantized scratch buffer; dequantize q8_0 with packed types (#27370); dequant only for large batches (#27438).
- **Vulkan**: Tiled transpose for 0↔2 permuted CONT (#26585) — 84% faster DeepSeek V4 prefill on RDNA; dequant q8_0 KV once in coopmat1 (#25494); shader source groups (#26666); Intel Xe SLM padding/reshape for coopmat mul_mm (#25380).
- **SYCL**: FWHT (Fast Walsh-Hadamard Transform) kernel (#27298) — 3.3–6x speedup; OPT_STEP_ADAMW/SGD ops (#25268); Q2_K MMVQ+ESIMD kernels (added then reverted); Q5_K ESIMD kernel (#26376); Alchemist GPU oneDNN gate logic (#26635); warning fixes (#26713).
- **OpenCL (Adreno)**: MoE per-expert bias fusion (#26431); SSM_SCAN kernel for Mamba-2 (#26439); deterministic MoE expert scatter (#26464); Adreno A7X compiler SIGSEGV workaround for mixed-type FA programs; vocab-scale K-quant lm_head kept on CPU for A7X (#26440).
- **Hexagon**: FA HMX queue fix and D matrix packing (#27042).
- **WebGPU**: MulMat with overlapping src0/src1 for MiniMax-01 (#27321).
- **Server UI**: Major multi-PR refactoring — stores split into domain namespaces (#27240), services consolidated (#27239), stores consolidated (#27238); settings navigation cleanup (#27241); built-in tools renamed to server/browser tools (#27271); `get_datetime` tool moved to frontend (#27255); browser `get_info` tool added (#27251); MCP tool result `structuredContent` support (#26691); API key field masking (#26562); settings persistence ordering fix (#27365); alphabetical enum ordering enforced (#27272).
- **Converters**: `@ModelBase.example` decorator for model registration (#27208); speculators-format DSpark (#26275); Kimi K3 MXFP4 repack (#26185); BailingMoE3 Q-LoRA (#26608); Nemotron 3 Ultra fix (#27101).
- **Quantization**: Memory usage optimization — weights are evicted from memory after processing each layer (#22877), reducing peak RSS during quantization.
- **Other**: ggml `__fp16` gated on `__ARM_FP16_FORMAT_IEEE` for 32-bit ARM (#26860); RPC use_count populated for fusion (#27142); multimodal SHA-256 hashing (#27274); LFM2 non-tiled thumbnail skip (#27246); Granite preprocessor hardened (#27235); chat format refactoring for string/typed content (#27130); `ggml_concat` usage reduced (#27176); duplicate metadata load removed (#27378).
