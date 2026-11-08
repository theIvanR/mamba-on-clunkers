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

5. **Windows NCCL Support**
   - Backported and patched NCCL to enable multi-GPU communication on Windows.
   - Enabled the underlying communication support required for Distributed Data Parallel (DDP) training across multiple legacy GPUs.
## 1. Reference build configuration

The following configuration is the reference environment used for development and testing.

| Component | Version / configuration |
|---|---|
| PyTorch | 2.7.1 |
| Causalconv 1D | 1.7.0 |
| Python | 3.11 |
| CUDA Toolkit | 11.8 |
| cuDNN | 8.7.0 |
| Visual Studio | 2019 / MSVC v142 |
| Intel oneAPI | 2024.2.1.101 |
| NVIDIA driver | 472.50 |
| Target architectures | 3.5; 3.7; 5.0; 5.2; 6.0; 6.1; 7.0; 7.5 |

The goal is to retain support for legacy GPU architectures while running a modern PyTorch stack on Windows.

**Build architecture discovery**

Unlike conventional CUDA extension build configurations, this project does not require CUDA architectures to be manually maintained in `setup.py`.

Instead, the build script:

1. Initializes the Visual Studio compiler environment.
2. Selects the intended Python environment.
3. Queries the installed PyTorch installation using `torch.cuda.get_arch_list()`.
4. Converts the reported architecture identifiers into `TORCH_CUDA_ARCH_LIST`.
5. Passes the resulting architecture list to the extension build.

Consequently, the architecture targets are derived from the actual PyTorch installation rather than being independently hardcoded in the package configuration.

`setup.py` is responsible for interpreting the supplied configuration and compiling the extension accordingly. The build script determines the environment and the targets.

## 2. High-performance prebuilt wheels
- for high performance Pytorch 2.7.1: https://github.com/theIvanR/torch-on-clunkers
- for causal-conv1d: https://drive.google.com/file/d/1deRG1heG6vdTK5LQpDq4JaMvartLdbx9/view?usp=drive_link
- mamba-ssm: https://drive.google.com/file/d/1M00K9hdru_qfOX80KEqfau67DiI7AF9I/view?usp=drive_link

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
