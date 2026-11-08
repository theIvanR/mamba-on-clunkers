# Copyright (c) 2023, Albert Gu, Tri Dao.
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version

from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

PACKAGE_NAME = "mamba_ssm"

# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI.
# No-op on Windows (MSVC has no libstdc++ dual-ABI), harmless elsewhere.
FORCE_CXX11_ABI = os.getenv("MAMBA_FORCE_CXX11_ABI", "FALSE") == "TRUE"


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_ver = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_ver


def append_nvcc_threads(nvcc_extra_args):
    return nvcc_extra_args + ["--threads", "4"]


def get_arch_flags_from_env():
    """
    Build -gencode flags strictly from TORCH_CUDA_ARCH_LIST.

    Accepted formats:
      "7.5;8.0;8.6"
      "7.5 8.0 8.6"
      "7.5,8.0,8.6"
      "7.5+PTX;8.0;8.6"
      "9.0a"   (feature suffix pass-through)

    No defaulting, no auto-detection: if the env var is missing/empty/invalid,
    this raises and the build hard-fails.
    """
    raw = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
    if not raw:
        raise RuntimeError(
            "TORCH_CUDA_ARCH_LIST must be set explicitly to a non-empty list of CUDA "
            "architectures (e.g. '7.5;8.0;8.6' or '7.5 8.0 8.6+PTX'). "
            "Refusing to build without an explicit architecture list."
        )

    entries = [tok for tok in re.split(r"[;,\s]+", raw) if tok]
    if not entries:
        raise RuntimeError(
            f"TORCH_CUDA_ARCH_LIST={raw!r} did not contain any usable architectures."
        )

    flags = []
    for entry in entries:
        arch = entry
        ptx = False
        if arch.endswith("+PTX"):
            ptx = True
            arch = arch[: -len("+PTX")]

        # "9.0" -> digits="90", suffix=""  ;  "9.0a" -> digits="90", suffix="a"  ;  "10.0" -> "100"
        m = re.match(r"^(\d+)\.(\d+)([a-zA-Z]*)$", arch)
        if not m:
            raise RuntimeError(
                f"Invalid CUDA architecture entry {entry!r} in "
                f"TORCH_CUDA_ARCH_LIST={raw!r}"
            )
        digits = m.group(1) + m.group(2)
        suffix = m.group(3)

        flags.append("-gencode")
        flags.append(f"arch=compute_{digits}{suffix},code=sm_{digits}{suffix}")
        if ptx:
            flags.append("-gencode")
            flags.append(f"arch=compute_{digits}{suffix},code=compute_{digits}{suffix}")

    return flags


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))

if CUDA_HOME is None:
    raise RuntimeError(
        f"{PACKAGE_NAME} is built from source only and requires nvcc. "
        "CUDA_HOME is not set / nvcc not found on PATH. "
        "If you're inside a PyTorch docker image, only 'devel' images ship nvcc."
    )

_, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
if bare_metal_version < Version("11.6"):
    raise RuntimeError(
        f"{PACKAGE_NAME} is only supported on CUDA 11.6 and above.  "
        "Note: make sure nvcc has a supported version by running nvcc -V."
    )

cc_flag = get_arch_flags_from_env()
print("Building with CUDA arch flags:")
for tok in cc_flag:
    print("   ", tok)

if FORCE_CXX11_ABI:
    torch._C._GLIBCXX_USE_CXX11_ABI = True

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": append_nvcc_threads(
        [
            "-O3",
            "-std=c++17",
            "-D_USE_MATH_DEFINES",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "--ptxas-options=-v",
            "-lineinfo",
        ]
        + cc_flag
    ),
}

ext_modules = [
    CUDAExtension(
        name="selective_scan_cuda",
        sources=[
            "csrc/selective_scan/selective_scan.cpp",
            "csrc/selective_scan/selective_scan_fwd_fp32.cu",
            "csrc/selective_scan/selective_scan_fwd_fp16.cu",
            "csrc/selective_scan/selective_scan_fwd_bf16.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp32_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_fp16_complex.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_real.cu",
            "csrc/selective_scan/selective_scan_bwd_bf16_complex.cu",
        ],
        extra_compile_args=extra_compile_args,
        include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
    )
]


def get_package_version():
    with open(Path(this_dir) / PACKAGE_NAME / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("MAMBA_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)


setup(
    name=PACKAGE_NAME,
    version=get_package_version(),
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
            "mamba_ssm.egg-info",
        )
    ),
    author="Tri Dao, Albert Gu, Ivan Rodionov",
    author_email="",
    description="Mamba state-space model",
    url="https://github.com/theIvanR/mamba-on-clunkers",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
        # "triton>=3.5.0",
        "transformers",
        "tilelang==0.1.8",
        "apache-tvm-ffi<=0.1.12",
        "quack-kernels>=0.3.4",
        # "causal_conv1d>=1.4.0",
    ],
)