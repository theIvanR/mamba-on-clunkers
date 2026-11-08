import torch
import time
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm.utils.generation import InferenceParams

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def test_numerical_correctness():
    print_header("Test 1: Numerical Correctness (CUDA vs PyTorch Ref)")
    
    batch, dim, seqlen, dstate = 2, 64, 128, 16
    device = "cuda"
    dtype = torch.float32

    u = torch.randn(batch, dim, seqlen, device=device, dtype=dtype)
    delta = torch.rand(batch, dim, seqlen, device=device, dtype=dtype) * 0.1
    A = -torch.rand(dim, dstate, device=device, dtype=dtype) - 1.0
    B = torch.randn(batch, dstate, seqlen, device=device, dtype=dtype)
    C = torch.randn(batch, dstate, seqlen, device=device, dtype=dtype)
    D = torch.randn(dim, device=device, dtype=dtype)

    # CUDA Kernel
    out_cuda = selective_scan_fn(u, delta, A, B, C, D, delta_softplus=True)
    
    # PyTorch Reference
    out_ref = selective_scan_ref(u, delta, A, B, C, D, delta_softplus=True)

    diff = (out_cuda - out_ref).abs().max().item()
    print(f"  Max absolute error: {diff:.6e}")
    
    if diff < 1e-4:
        print("  [+] PASS: CUDA kernel matches PyTorch reference.")
    else:
        print("  [-] FAIL: Numerical mismatch detected!")

def test_sequence_lengths():
    print_header("Test 2: Sequence Length Edge Cases (Kernel Specializations)")
    
    device = "cuda"
    dtype = torch.float16
    dim, dstate = 128, 16
    
    # These specific lengths trigger different template instantiations in selective_scan_fwd_cuda
    lengths = [64, 128, 256, 512, 1024, 2048]
    
    for seqlen in lengths:
        try:
            model = Mamba(d_model=dim, d_state=dstate, d_conv=4, expand=2).to(device=device, dtype=dtype)
            x = torch.randn(2, seqlen, dim, device=device, dtype=dtype)
            
            start = time.time()
            out = model(x)
            torch.cuda.synchronize()
            end = time.time()
            
            print(f"  seqlen={seqlen:<5} | Shape: {out.shape} | Time: {(end-start)*1000:.2f} ms | [+] PASS")
        except Exception as e:
            print(f"  seqlen={seqlen:<5} | [-] FAIL: {e}")

def test_dtypes():
    print_header("Test 3: Data Types (FP32, FP16, BF16)")
    
    batch, seqlen, dim = 2, 128, 128
    device = "cuda"
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        # BF16 requires Ampere (sm_80) for hardware acceleration in cuBLAS
        if dtype == torch.bfloat16 and cc_major < 8:
            print(f"  dtype={str(dtype):<15} | [~] SKIP: Requires Compute Capability 8.0+ (cuBLAS limitation)")
            continue
            
        try:
            model = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2).to(device=device, dtype=dtype)
            x = torch.randn(batch, seqlen, dim, device=device, dtype=dtype, requires_grad=True)
            
            out = model(x)
            loss = out.sum()
            loss.backward()
            
            grad_mean = x.grad.mean().item()
            print(f"  dtype={str(dtype):<15} | Grad mean: {grad_mean:.4f} | [+] PASS")
        except Exception as e:
            print(f"  dtype={str(dtype):<15} | [-] FAIL: {e}")

def test_autoregressive_step():
    print_header("Test 4: Autoregressive Generation (State Update Kernel)")
    
    batch, dim = 1, 128
    device = "cuda"
    dtype = torch.float16
    
    try:
        model = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2).to(device=device, dtype=dtype)
        # Standalone Mamba blocks need layer_idx set for the state cache to work
        model.layer_idx = 0
        
        # Initialize state
        inference_params = InferenceParams(max_seqlen=2048, max_batch_size=batch)
        
        # Prefill: process the first 10 tokens in parallel
        prefill = torch.randn(batch, 10, dim, device=device, dtype=dtype)
        out_prefill = model(prefill, inference_params=inference_params)
        inference_params.seqlen_offset += 10
        
        print(f"  Prefill output: {out_prefill.shape}")
        
        # Step: generate tokens one at a time
        for i in range(5):
            token = torch.randn(batch, 1, dim, device=device, dtype=dtype)
            out_step = model(token, inference_params=inference_params)
            inference_params.seqlen_offset += 1
            
        print(f"  Step output:    {out_step.shape}")
        print("  [+] PASS: Autoregressive step (state passing) successful.")
    except Exception as e:
        print(f"  [-] FAIL: Step generation failed.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("MAMBA-ON-CLUNKERS TEST SUITE")
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)
        
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    test_numerical_correctness()
    test_sequence_lengths()
    test_dtypes()
    test_autoregressive_step()
    
    print("\nTesting complete!")