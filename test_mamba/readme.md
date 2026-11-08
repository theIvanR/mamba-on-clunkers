# Mamba-on-Clunkers Test Suite

This directory contains two testing scripts designed to verify the installation, numerical correctness, and runtime performance of `mamba_ssm` on legacy NVIDIA CUDA architectures (Kepler, Maxwell, and Pascal).

Ensure the required dependencies are installed:

```bash
pip install -r requirements.txt
```

At minimum, the tests require `torch` and `einops`. Python's `urllib.request` is part of the standard library and does not require a separate installation.

---

## 1. `test_mamba_basic.py` — Sanity Check

A fast (~30-second), comprehensive test of the custom CUDA kernels. It verifies that the compiled `.pyd` extension is executing correctly on your specific GPU architecture.

### What it tests

1. **Numerical Correctness:** Compares the custom CUDA `selective_scan_fn` implementation against the PyTorch reference implementation to verify numerical parity.
2. **Sequence Lengths:** Tests sequence lengths ranging from `64` to `2048`, covering the hardcoded template specializations in the C++ dispatch implementation.
3. **Data Types:** Tests both `FP32` and `FP16`. (`BF16` is automatically skipped on Kepler, Maxwell, and Pascal, where the required cuBLAS support is unavailable.)
4. **Autoregressive Step:** Tests the `selective_scan_update` kernel used for token-by-token autoregressive generation.

### How to run

```bash
python test_mamba_basic.py
```

### Expected Output

Successful execution should produce a `[+] PASS` message for each test, concluding with:

```text
Testing complete!
```

If a kernel fails, the script prints a traceback containing the corresponding CUDA error.

---

## 2. `test_mamba_llm.py` — End-to-End Training

A more demanding, real-world integration test. It downloads the TinyShakespeare dataset and trains a small 1.77M-parameter Mamba-1 language model from scratch.

This verifies the stability of the entire training stack, including the forward pass, backward pass, and optimizer, over hundreds of training iterations.

### What it does

1. Downloads the TinyShakespeare dataset if it is not already present locally.
2. Constructs a four-layer Mamba-1 model with the following configuration:
   - `d_model = 256`
   - `d_state = 16`
3. Trains the model for 500 steps using AdamW and a cosine learning rate schedule.
4. Generates a 100-character Shakespeare sample after training.

### How to run

```bash
python test_mamba_llm.py
```

### Expected Performance

On an NVIDIA Tesla K40c (Kepler), a typical run takes approximately **20 minutes**, averaging around **2.4 seconds per training step**.

The training loss should decrease from approximately `4.30` (random initialization baseline) to around `1.50` by the end of training.

Example output:

```text
Starting training...

Step | Loss   | LR      | Time
----------------------------------------
   0 | 4.3047 | 6.00e-06 | 1.7s
  50 | 2.3977 | 3.00e-04 | 2.3s
 ...
 499 | 1.5084 | 3.00e-05 | 2.3s

--- Sample Generation ---

ROMEO: No. O,ean my havvenred, that up, that heartimence
she ygate? Take light taken gviek fine but a virt
```

The generated text is intentionally shown as a raw sample; the objective is to verify successful end-to-end execution rather than language quality.

---

## Troubleshooting

### `RuntimeError: no kernel image is available for execution on the device`

The compiled extension does not contain compatible SASS machine code for your GPU architecture.

Check your `TORCH_CUDA_ARCH_LIST` configuration and rebuild the extension with the appropriate architecture targets.

### `ModuleNotFoundError: No module named 'triton'`

You are likely running an older version of this fork. Update to the latest commit, which includes the Triton fallback implementation.

### Out of Memory (OOM)

If your GPU has less than 4 GB of VRAM, try reducing `batch_size` or `block_size` in `test_mamba_llm.py`.

---

**Purpose:** These tests provide a practical verification that Mamba-1 can execute and train on legacy NVIDIA GPUs, including architectures that are no longer supported by many modern CUDA extensions.
