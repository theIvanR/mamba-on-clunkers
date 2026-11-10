// selective_scan_bwd_all.cu
#include "selective_scan_bwd_kernel.cuh"

// fp16, fp32 acc (real, complex)
template void selective_scan_bwd_cuda<at::Half, float>(SSMParamsBwd&, cudaStream_t);
template void selective_scan_bwd_cuda<at::Half, complex_t>(SSMParamsBwd&, cudaStream_t);

//bf16, fp32 acc (real, complex)
template void selective_scan_bwd_cuda<at::BFloat16, float>(SSMParamsBwd&, cudaStream_t);
template void selective_scan_bwd_cuda<at::BFloat16, complex_t>(SSMParamsBwd&, cudaStream_t);

//fp32, fp32 acc (real, complex)
template void selective_scan_bwd_cuda<float, float>(SSMParamsBwd&, cudaStream_t);
template void selective_scan_bwd_cuda<float, complex_t>(SSMParamsBwd&, cudaStream_t);