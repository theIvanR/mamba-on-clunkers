# Mamba SSM for legacy cuda devices on windows

A Windows-native implementation of the Mamba State Space Model (Mamba-1), adapted to support legacy NVIDIA GPU architectures within a modern PyTorch environment. The project focuses on retaining native CUDA support for older GPUs—including Kepler, Maxwell, Pascal, and Turing—while running cutting-edge AI models.

> Based on the work of Tri Dao and Albert Gu

## Project Roadmap

- **Phase 1: Implementation** <badge>Complete</badge>
  - Stable, pure-CUDA Mamba-1 execution across legacy GPU architectures (SM 3.5 through SM 7.5) on Windows.
- **Phase 2: Optimization & Profiling** <badge color="warning">In Progress</badge>
  - Kepler/Pascal-specific kernel tuning, shared-memory optimization, and atomic reduction improvements.
- **Phase 3: Cleanup** <badge color="secondary">Planned</badge>
  - Complete removal of unused Triton, TileLang, and ROCm/HIP code from the repository, resulting in a lean, pure-CUDA fork.

## Key Achievements

Porting a 2024-era architecture to 2012 hardware using a Windows/MSVC toolchain required addressing compatibility issues throughout the modern ML software stack. Here is what this fork actually delivers:

### Phase 1
1. **Triton Dependency Excision & Graceful Fallback**
   - Removed Triton from the core dependency requirements.
   - Implemented graceful fallbacks for Mamba-1 using PyTorch and custom CUDA kernels.
   - When Triton is unavailable, `mamba_ssm` can fall back to the non-Triton Mamba-1 execution path without failing during package initialization.
   - Mamba-2 and Mamba-3 remain outside the supported scope of this legacy CUDA fork.

2. **MSVC Compiler Compatibility**
   - Patched CUDA source code to comply with MSVC's stricter C++17 compilation requirements.
   - Addressed POSIX-specific `M_LOG2E` constants, `constexpr` lambda captures, and preprocessor directives inside template lambdas.

3. **Dynamic Legacy Architecture Targeting**
   - Replaced upstream's hardcoded Ampere/Hopper architecture assumptions.
   - The build system queries `torch.cuda.get_arch_list()` and dynamically configures the corresponding SASS/PTX compilation targets.
   - Supports explicit targeting of legacy architectures, including Kepler (SM 3.5).

4. **Numerical Parity**
   - Achieved close numerical agreement with the PyTorch reference implementation on a 2013 NVIDIA Tesla K40c.
   - Maximum absolute error: `3.8e-06`.

## Phase 2 — Optimization & Profiling

Phase 2 focuses on tuning the Mamba-1 kernels for legacy NVIDIA architectures. Kepler's register-file size, warp-scheduler behavior, and L2 atomic throughput differ substantially from the Ampere/Hopper hardware upstream was tuned for. Predictions from Ampere intuition don't transfer — every change below was measured, and several didn't pan out.

### Completed optimizations

- **`checkpoint_lvl` exposed as a constructor parameter**
  - Upstream hardcodes `checkpoint_lvl=1` in `Mamba.forward`, forcing `conv1d_out` and `delta` recomputation in the backward pass.
  - On cards with 12GB+ VRAM and small models (<2M params), the memory savings are irrelevant — the two cached tensors total ~32MB/layer at `seqlen=256`.
  - Exposed as `Mamba(checkpoint_lvl=0)`. **Measured win: 1.4% step time** on K40c.

- **Redundant delta reload removed in `selective_scan_bwd_kernel.cuh`**
  - The `kDeltaSoftplus` epilogue re-loaded the same `delta` bytes into a shadowed register array that the chunk prologue had already loaded and left untouched.
  - Removed one `load_input<Ktraits>(delta, ...)` call and one `__syncthreads()` barrier per chunk.
  - **Measured: bwd kernel time unchanged** — nvcc `-O3` was likely already hoisting the redundant load. Retained as a code-quality improvement, not a perf win.

- **`kMinBlocks` raised from 2 → 3 for the bwd kernel (`kNThreads=32`)**
  - Kepler SM has 64 warp slots. The bwd kernel at `kMinBlocks=2` achieved 3.1% occupancy; at 3, that becomes 4.7%.
  - Register budget at 3 is `65536 / (32 × 3) = 682`, well above the 255 hardware cap — no register reduction or spilling.
  - Conservative bump pending the full sweep (see "In progress").

### Measured impact

Profiler-verified on a Tesla K40c, batch=8, seqlen=256, FP32, 4-layer Mamba-1 LM (~1.77M params):

| Kernel | Before | After | Δ |
|---|---|---|---|
| `selective_scan_fwd_kernel` (per layer) | 4.10 ms | 3.49 ms | **−15%** |
| `causal_conv1d_fwd` | 584 µs | 472 µs | −19% |
| `MambaInnerFnBackward` (4 layers aggregate) | 15.33 ms | 15.42 ms | unchanged |
| **Total self-CUDA time** | **46.36 ms** | **42.44 ms** | **−8.5%** |

The forward-kernel gains are real and reproducible. The backward kernel did not move much despite the delta reload fix.

### In progress

- **`kMinBlocks` sweep**
  - Predicted optimum for Kepler: `fwd=8` (budget 256, occupancy 12.5% vs 7.8% at 5), `bwd=6` (budget 341, occupancy 9.4% vs 4.7% at 3).
  - Tooling: `sweep_kminblocks.py`. **Expected additional win: 3–8%.**

- **Batch size throughput sweep**
  - Bumping batch from 8 → 16 → 32 amortizes the ~15% CPU launch overhead across more tokens.
  - Does not reduce per-step time. **Expected: +10–15% tokens/sec.**

### Negative results (documented for future reference)

- **`checkpoint_lvl=0` alone: 1.4%, not the predicted 10–15%.**
  - Recomputing `conv1d_out` and `delta` in backward is cheaper than expected on Kepler — working sets are small (~512 bytes/block) and L2 absorbs most of the reload cost.

- **Delta reload fix: no measurable bwd kernel change.**
  - nvcc `-O3` was likely hoisting the load. The fix removes misleading code, not a bottleneck.

- **Double-buffering `smem_load` in the forward kernel: zero benefit.**
  - For `seqlen ≤ 256` with `kNThreads × kNItems = 256`, `n_chunks = 1` — each block loads each tensor exactly once. No next chunk to overlap with.

- **One big win remains out of reach.**
  - The 16-iteration serial state loop in `selective_scan_bwd_kernel` is the single biggest remaining bottleneck (~40–50% of bwd kernel time).
  - Parallelizing it across warps could yield 15–25% on the bwd kernel, but requires rewriting the shared-memory layout for the scan operators, parallelizing `BlockScan`/`BlockReverseScan` across the state dimension, and carries high risk of numerical drift from changed accumulation order.
  - Will be implemented at a later time.
 
## 2. High-performance prebuilt wheels
- for high performance Pytorch 2.7.1: https://github.com/theIvanR/torch-on-clunkers
- for causal-conv1d: https://drive.google.com/file/d/1deRG1heG6vdTK5LQpDq4JaMvartLdbx9/view?usp=drive_link
- mamba-ssm: 

> For additional Kepler GPU performance modifications, see [Kepler on Steroids](https://drive.google.com/file/d/1WeO1uz6DdOiAQooTUG2ouHa9ajZYDG0p/view?usp=drive_link).

## 3. Prerequisites

The reference development environment uses:

- Git
- Miniconda
- Visual Studio 2019 with the MSVC v142 C++ toolchain
- NVIDIA CUDA Toolkit 11.8
- NVIDIA driver 472.50
- PyTorch 2.7.1 (see link above)
- Causal conv 1d 1.7.0 (link above)
- Intel oneAPI 2024.2.1.101
- cuDNN 8.7.0

**Important:** For legacy NVIDIA driver compatibility, install the CUDA Toolkit without replacing your existing display driver. Verify that the installed CUDA toolkit, driver and cuDNN combination is compatible before proceeding.

Install cuDNN into the appropriate CUDA directories according to its Windows installation instructions.

The CUDA compiler (`nvcc`) and Microsoft C++ compiler (`cl`) must both be available to the build process.

## 4. Prepare the build environment

Create and activate a dedicated Python 3.11 environment:

```bat
conda create -n mamba_legacy python=3.11 -y
conda activate mamba_legacy
```

Install the prebuilt PyTorch and causal-conv1d wheels provided above.

```bat
python -m pip install --no-deps torch-2.7.1-cp311-cp311-win_amd64.whl
python -m pip install --no-deps causal_conv1d-1.7.0-cp311-cp311-win_amd64.whl
```

Verify that both packages are available:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python -c "import causal_conv1d; print('causal-conv1d:', causal_conv1d.__version__)"
```

Before proceeding, ensure that:

- The environment uses Python 3.11.
- PyTorch and causal-conv1d import successfully.
- The CUDA Toolkit 11.8 compiler (`nvcc`) is available.
- The Visual Studio 2019 x64 compiler environment (`cl`) is initialized.
- The installed PyTorch build exposes the intended CUDA architectures.

**Important:** The supplied wheels are built for the reference environment described in Section 1. Use compatible wheels and dependencies rather than allowing pip to replace the preconfigured PyTorch installation.

## 5. Build the extension

Clone the repository:

```bat
git clone https://github.com/theIvanr/mamba-on-clunkers
cd mamba-on-clunkers
```

Run the Windows build script:

```bat
build_cuda_win.bat
```

The script performs the following operations:

- Initializes the Visual Studio 2019 x64 compiler environment.
- Selects the configured Python interpreter.
- Verifies the PyTorch and CUDA environment.
- Checks for the availability of MSVC and NVCC.
- Detects the CUDA architectures compiled into the installed PyTorch build.
- Exports those architectures through `TORCH_CUDA_ARCH_LIST`.
- Builds a distributable wheel using `pip wheel`.

The build uses:

```bat
python -m pip wheel . --no-build-isolation --no-deps --no-cache-dir -w dist -v
```

This produces a wheel in the `dist` directory without installing the package or modifying the active Python environment.

The architecture selection is handled by the build script; `setup.py` simply consumes the supplied configuration.

## 7. Install the wheel

After a successful build, install the generated wheel:

```bat
python -m pip install --no-deps dist\mamba_ssm-2.3.2.post1-cp311-cp311-win_amd64.whl
```

OPTIONAL: Run the test suite to verify the extension.

```bat
python test_mamba.py
```

## AMD? 
- coming later
