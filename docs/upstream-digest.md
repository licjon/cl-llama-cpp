# llama.cpp Upstream Digest

Pin at last update: 4988f6e866057afd130c1515ecef0c9bab9a15f8
Last covered upstream commit: d7bd3bfcad3e29c7e49fd26f38c79ee3e9a3fd6b

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

---

## 2026-08-28 — 108 commits since last digest

### New Features

- **Qwen3.8-Flash-Next (qwen4exp) architecture** (#27742): A large new hybrid model from Alibaba combining gated delta net (recurrent) layers with full attention layers, hyper-connections (wide residual stream), MoE with gated shared experts, QSA sparse attention with block-level top-k indexing, and a PLE (per-layer embedding) n-gram hash table. Includes quantized KV cache support, tensor parallelism, multi-slot server support, and multimodal image handling. This is a substantial architecture addition (~1200 lines of model code).
- **DFlash2 speculative decoding** (#27816): A new speculative decoding method that uses local convolution for candidate selection instead of a separate draft model, offering a different speed/quality tradeoff from DFlash1 and DSpark.
- **DSpark support for Nemotron3.5** (#27804) and **BailingMoE3** (#27508): Extends the DSpark speculative decoding method to two more model families.
- **dots3-note vision+audio** (#27524): Multimodal support (both image and audio input) for the dots3-note model.
- **nanbeige4.2-3B** (#27730): New model support for the Nanbeige 4.2 3B architecture.
- **MTP for GLM-4.5-Air** (#26534): Multi-token prediction speculative decoding added to the GLM-4.5-Air model.
- **Lazy tensor reading** (#27794): New `LLAMA_LAZY_MODE` option that reads large tensor rows on demand via mmap instead of loading them upfront, reducing memory usage for models with very large gather tables (e.g. qwen4exp's 26 GiB PLE table).
- **`--video-*` CLI arguments** (#24318): New CLI flags for video input in multimodal models, plus a `mtmd_helper_init_opt` API function.
- **`--n-cpu-ffn` option** (#26622): CPU-offload dense FFN weights for the first N layers, complementing the existing `--n-cpu-moe`.
- **`--kv-unified-per-slot`** (#24124): Server option controlling how many KV cache slots each client connection gets in unified cache mode.
- **Token ID tracking in KV cells** (#27762): KV cache cells now record which token ID produced them, enabling `get_prev_tokens()` for n-gram history lookups without separate bookkeeping.
- **Synthetic speculative acceptance** (#27711): Benchmark-only option to simulate speculative decoding acceptance rates without an actual draft model, useful for measuring overhead.
- **RPC async backend APIs** (#18626): The RPC (remote procedure call) backend now supports event-driven and asynchronous tensor operations, reducing latency for distributed inference.
- **Apple RDMA as RPC transport** (#26421): On Apple Silicon Macs, the RPC backend can now use RDMA for low-latency inter-node communication instead of TCP sockets.
- **`LLAMA_SERVER_SLOTS_N_DIFF` env var** (#27600): Controls the number of slot differences the server tracks.
- **llama.cpp version 0.3.0** (#27696): Version bump with semantic versioning enforced. ggml bumped to 0.22.0.
- **Metal per-op source split + parallel compile** (#26561): Metal shader sources are split per-operation and compiled in parallel, significantly reducing Metal backend initialization time.
- **Metal per-device flash-attention tuning** (#26570): Flash-attention vector kernels are now tuned per Apple GPU model (M1 Pro through M5 Max) for optimal (Q, NE) tile dispatch, with an offline tuning tool.

### Bug Fixes

- **Vulkan view-alias dependency ordering** (#27812): The Vulkan graph optimizer failed to treat two views of the same tensor as dependent, silently reordering reads and writes across aliased state. This produced wrong tokens under greedy decoding with no error logged, hitting recurrent-state models (like Qwen3.8) on AMD and NVIDIA Vulkan.
- **conv_transpose_2d multi-batch** (#26132): Only the first batch was computed in transposed 2D convolution; all subsequent batches were left as zeros. Fixed on both CPU and Metal.
- **DeepSeekV4 rollback with multi-sequence** (#26756): Cache rollback in multi-sequence mode was incorrect, causing state corruption.
- **ggml_clamp** (#27644): The clamp operation produced incorrect results in certain cases.
- **Metal OOM crash** (#25371): A null Metal buffer allocation (e.g. out-of-memory on iOS) caused a hard crash (EXC_BAD_ACCESS) instead of a recoverable error.
- **Metal memory leaks** (#27758): Missing autorelease pools caused memory leaks during Metal operations.
- **Meta tensor split state propagation** (#27574): Tensor-parallel mode on the Meta backend dropped split state, causing assertion failures.
- **Grammar `\-` in character classes** (#27591): The GBNF grammar parser rejected `\-` (escaped hyphen) in character classes even though the grammar generator produced it, breaking tool-call grammars.
- **Server: prefilled assistant + tool calls** (#27626): When `--prefill-assistant` was enabled and the last assistant message contained tool calls, the tool calls were silently stripped. Now rejects this combination with a clear error.
- **Draft-MTP with embeddings** (#27400): MTP speculative decoding crashed when the input batch used embeddings instead of token IDs.
- **Video moov atom parsing** (#27596): Videos with the moov atom at the end of the file failed to load in multimodal processing.
- **Nemotron 3.5 Lightning conversion** (#27729): The converter miscounted attention layers when using transformers >= 5.6, producing incorrect head_count_kv metadata.
- **GLM conversion regression** (#27655): A regression in index_tensors broke GLM model conversion.
- **Mamba-2 single-token GEMV** (#27513): Mamba-2 in/out projections dispatched GEMV (one row at a time) instead of GEMM (batched), hurting prefill performance. Now flattened for batch dispatch.
- **KV cache whole-context restore** (#qwen4exp series): Restoring a whole-context state with multiple streams cleared already-restored streams on each iteration, losing all but the last sequence.
- **Quantizer memory** (#27795, qwen4exp series): The quantizer's working buffer was sized at `nelements * 4` bytes regardless of target type, wasting ~150 GB on models with very large tensors. Now sized exactly, and large tensors are processed in row bands capped at 1 GiB.
- **Various backend fixes**: Vulkan warp-size clamping (#27726), SYCL tq2_0 unsupported marking (#27660), WebGPU infinity handling in ARGSORT/TOP_K (#27538), Hexagon RMS_NORM_MUL weight-offset for grouped norms (#27798), OpenCL binary kernel additions (#27768).

### Capability Gaps

All previously identified breaking API changes (from the 2026-08-15 digest) remain unresolved in `src/bindings.lisp`:

- **`llama_sampler_init_penalties`** still uses the old 4-argument signature (upstream now requires `n_vocab` as the first argument).
- **`llama_sampler_init_dry`** still passes `n_ctx_train` (upstream removed it).
- **`llama_model_params` struct** still has `use_mmap`/`use_direct_io`/`use_mlock` bools (upstream replaced them with `load_mode` enum and added `load_mtp`).
- **`llama_context_params` struct** is missing the `n_outputs_max_per_seq` field.
- **`llama_version()`**, **`llama_ftype_name()`**, **`llama_model_ftype()`**, **`llama_load_mode_name()`/`llama_load_mode_from_str()`**, **`llama_model_n_layer_nextn()`**, **`llama_vocab_get_suppress_tokens()`**, **`llama_sampler_copy()`** — all still unbound.
- **`llama_load_mode` enum** and **`LLAMA_FTYPE_MOSTLY_Q2_0`** — still not defined.

New gaps introduced in this batch:

- **BREAKING — `llama_model_params` struct layout changed again**: A new `lazy_mode` field (enum `llama_lazy_mode`) was inserted after `load_mode`, further shifting all subsequent field offsets. The bindings' struct definition is now two fields behind upstream (`load_mode` + `lazy_mode` replacing three bools, plus the new insertion).
- **BREAKING — `llama_model_quantize_params` struct layout changed**: A new `max_buf_size` field (`size_t`) was appended, changing the struct size. Code passing this struct by value may read uninitialized memory for the new field.
- **New enum `llama_lazy_mode`**: Values `LLAMA_LAZY_MODE_OFF` (0), `LLAMA_LAZY_MODE_AUTO` (1), `LLAMA_LAZY_MODE_ON` (2). Not defined in bindings.
- **`LLAMA_SESSION_VERSION` bumped to 10**, **`LLAMA_STATE_SEQ_VERSION` bumped to 3**: Session state files from newer llama.cpp builds are incompatible with older versions. Not reflected in bindings constants.

### Other / Internal

- **Build**: llama.cpp version 0.2.0 → 0.3.0; ggml 0.21.0 → 0.22.0; release workflow improvements; ccache moved to HF buckets (#27699); ccache-clear improvements (#27504, #27602); CI cache bucket made public (#27728).
- **CI**: Windows ARM64 ROCm DLL bundling (#26973); ROCm Ubuntu job restored (#27399) with content-based compiler check; Metal tensor-split test added (#27598); test-llama-archs now runs test-save-load-state across all architectures (#27755); UI build restructured — npm build disabled by default, prebuilt artifact reused (#27706); OpenVINO updated to 2026.3.1 (#27843).
- **CUDA**: MMQ unblocked for MoE on sm_60/Pascal (#26264); POOL_1D support (#27573).
- **Metal**: Per-op source split with parallel compilation (#26561) — 8→20 metal libraries compiled in parallel; per-device FA-vec tuning for M1 Pro, M2 Ultra, M3 Max, M4, M4 Pro, M5, M5 Max, M5 Pro (#26570, #27824, #27863, #27875); chunked SSD MMA for Mamba-2 prefill optimization (#26647); null-check OOM fix (#25371); memory leak fixes (#27758).
- **Vulkan**: LIGHTNING_INDEXER op for DSV4 (#27453); hoisted row IDs and expert count in shaders (#26686); cross_entropy_loss and back (#27216); PAD_REFLECT_1D (#26586); warp-size clamping for >64 warps (#27726); tiled transpose for permuted CONT.
- **SYCL**: Q2_K reordered MMVQ+ESIMD kernels re-added (#27490); quantized KV decode on BMG via TILE (#26689); F16 KV cache bind-in-place for oneDNN SDPA (#27468); tq2_0 marked unsupported (#27660).
- **Hexagon**: Multi-NPU support (IQ9, IQ10) with fully async backend (#26501) — non-host buffers, DMA pipelining, fence-based synchronization, ALLREDUCE with fused ADD, multi-device profiling; ABS and LOG unary ops (#27786).
- **OpenCL**: Binary kernels for MoE q4_0/mxfp4 dp4a (#27768).
- **RPC**: Event and async backend APIs (#18626); Apple RDMA transport (#26421).
- **KleidiAI**: Build system reworked (#26077) for ARM CPU acceleration.
- **Server UI**: Major overhaul — browser-style conversation tabs (#27263); settings and MCP servers moved to dialog-based views (#27744); per-conversation tool policy replacing MCP overrides (#27745); chat form actions UI/UX improvements (#27746); dialog component restyling (#27743); ESLint config updates (#27700).
- **Converters**: Nemotron-H LoRA GGUF fix (#27356); Nemotron 3.5 Lightning layer fix (#27729); GLM index_tensors regression fix (#27655); qwen4exp PLE streaming with LazyChunkedTensor; ndarray conversion guard (#27869).
- **Quantization**: Working memory cap to avoid loading huge tensors into RAM (#27795); output buffer sized exactly (#qwen4exp); row-band dequantize/quantize for >1 GiB tensors; tensor_type_fallback for 32-block types with odd ncols.
- **Other**: DeepSeek V4 tensor split (`-sm tensor`) support (#26490); MiniMax-01 graph simplification (#27790); ggml_clamp fix (#27644); concat op row-level memcpy optimization (#24575); JSON abstraction layer (#27511); repetition_penalty from generation_config.json (#27659); Pillow-accurate resize algorithm for multimodal (#27594); WebP via ffmpeg (#27520); ggml_rope_set_offset usage (#27521); subprocess.h update (#27409); common device_info loop skip when not printing (#26692); KV cache context-per-slot (#24124).
