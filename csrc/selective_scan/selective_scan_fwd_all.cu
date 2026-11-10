// selective_scan_fwd_all.cu
#include "selective_scan_fwd_kernel.cuh"

// fp16, fp32 acc (real, complex)
template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase&, cudaStream_t);
template void selective_scan_fwd_cuda<at::Half, complex_t>(SSMParamsBase&, cudaStream_t);

//bf16, fp32 acc (real, complex)
template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase&, cudaStream_t);
template void selective_scan_fwd_cuda<at::BFloat16, complex_t>(SSMParamsBase&, cudaStream_t);

//fp32, fp32 acc (real, complex)
template void selective_scan_fwd_cuda<float, float>(SSMParamsBase&, cudaStream_t);
template void selective_scan_fwd_cuda<float, complex_t>(SSMParamsBase&, cudaStream_t);