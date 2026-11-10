#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>

#include <cstdint>
#include <vector>

#include "selective_scan.h"

// ============================================================
// Dispatch macros (unchanged — this is the PyTorch-idiomatic way
// to dispatch on ScalarType; keep them.)
// ============================================================
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

#define DISPATCH_WTYPE_FLOAT_AND_COMPLEX(WTYPE, NAME, ...)                           \
    if (WTYPE == at::ScalarType::Float) {                                            \
       using weight_t = float;                                                       \
        __VA_ARGS__();                                                               \
    } else if (WTYPE == at::ScalarType::ComplexFloat) {                              \
        using weight_t = c10::complex<float>;                                        \
        __VA_ARGS__();                                                               \
    } else {                                                                         \
        AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
    }

// ============================================================
// CUDA kernel entry points (defined in *_all.cu)
// ============================================================
template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

template <typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream);

// ============================================================
// ScanConfig: resolved shape and flags, single source of truth.
// Computed once per call by validate_scan_inputs and consumed by
// both the params setters and the entry points.
// ============================================================
struct ScanConfig {
    int64_t batch, dim, seqlen, dstate, n_groups, n_chunks;
    bool is_variable_B, is_variable_C, is_complex, delta_softplus;
};

// Validates u/delta/A/B/C together and resolves the config.
// Fwd and bwd both call this, so shape/stride/device validation
// lives in exactly one place.
ScanConfig validate_scan_inputs(
        const at::Tensor& u, const at::Tensor& delta,
        const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
        at::ScalarType input_dtype, at::ScalarType weight_dtype,
        bool delta_softplus)
{
    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;
    const bool is_complex    = weight_dtype == at::ScalarType::ComplexFloat;

    TORCH_CHECK(delta.scalar_type() == input_dtype);
    TORCH_CHECK(B.scalar_type() == (is_variable_B ? input_dtype : weight_dtype));
    TORCH_CHECK(C.scalar_type() == (is_variable_C ? input_dtype : weight_dtype));

    for (const auto* t : {&u, &delta, &A, &B, &C}) {
        TORCH_CHECK(t->is_cuda());
    }
    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const int64_t batch    = sizes[0];
    const int64_t dim      = sizes[1];
    const int64_t seqlen   = sizes[2];
    const int64_t dstate   = A.size(1);
    const int64_t n_groups = is_variable_B ? B.size(1) : 1;
    const int64_t n_chunks = (seqlen + 2048 - 1) / 2048;

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    CHECK_SHAPE(u, batch, dim, seqlen);
    CHECK_SHAPE(delta, batch, dim, seqlen);
    CHECK_SHAPE(A, dim, dstate);
    if (is_variable_B) {
        CHECK_SHAPE(B, batch, n_groups, dstate, is_complex ? seqlen * 2 : seqlen);
        TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);
    } else {
        CHECK_SHAPE(B, dim, dstate);
    }
    if (is_variable_C) {
        CHECK_SHAPE(C, batch, n_groups, dstate, is_complex ? seqlen * 2 : seqlen);
        TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);
    } else {
        CHECK_SHAPE(C, dim, dstate);
    }

    return ScanConfig{batch, dim, seqlen, dstate, n_groups, n_chunks,
                      is_variable_B, is_variable_C, is_complex, delta_softplus};
}

// ============================================================
// Params setters.
//
// Contract: `p` is zero-initialized by the caller via `{}`.
// Setters write only the fields relevant to this call; fields for
// absent optional tensors (z, out_z, dz) remain zero.
// ============================================================

void set_ssm_params_fwd(SSMParamsBase& p, const ScanConfig& cfg,
                        const at::Tensor& u, const at::Tensor& delta,
                        const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
                        const at::Tensor& out,
                        const at::Tensor& z, const at::Tensor& out_z,
                        void* D_ptr, void* delta_bias_ptr, void* x_ptr,
                        bool has_z)
{
    // Scalars and flags
    p.batch             = static_cast<int>(cfg.batch);
    p.dim               = static_cast<int>(cfg.dim);
    p.seqlen            = static_cast<int>(cfg.seqlen);
    p.dstate            = static_cast<int>(cfg.dstate);
    p.n_groups          = static_cast<int>(cfg.n_groups);
    p.n_chunks          = static_cast<int>(cfg.n_chunks);
    p.dim_ngroups_ratio = static_cast<int>(cfg.dim / cfg.n_groups);
    p.is_variable_B     = cfg.is_variable_B;
    p.is_variable_C     = cfg.is_variable_C;
    p.delta_softplus    = cfg.delta_softplus;

    // Pointers
    p.u_ptr          = u.data_ptr();
    p.delta_ptr      = delta.data_ptr();
    p.A_ptr          = A.data_ptr();
    p.B_ptr          = B.data_ptr();
    p.C_ptr          = C.data_ptr();
    p.out_ptr        = out.data_ptr();
    p.D_ptr          = D_ptr;
    p.delta_bias_ptr = delta_bias_ptr;
    p.x_ptr          = x_ptr;
    p.z_ptr          = has_z ? z.data_ptr()     : nullptr;
    p.out_z_ptr      = has_z ? out_z.data_ptr() : nullptr;

    // Strides (elements, not bytes)
    p.A_d_stride      = A.stride(0);
    p.A_dstate_stride = A.stride(1);
    p.B_dstate_stride = cfg.is_variable_B ? B.stride(2) : B.stride(1);
    p.C_dstate_stride = cfg.is_variable_C ? C.stride(2) : C.stride(1);
    if (cfg.is_variable_B) {
        p.B_batch_stride = B.stride(0);
        p.B_group_stride = B.stride(1);
    } else {
        p.B_d_stride = B.stride(0);
    }
    if (cfg.is_variable_C) {
        p.C_batch_stride = C.stride(0);
        p.C_group_stride = C.stride(1);
    } else {
        p.C_d_stride = C.stride(0);
    }
    p.u_batch_stride     = u.stride(0);
    p.u_d_stride         = u.stride(1);
    p.delta_batch_stride = delta.stride(0);
    p.delta_d_stride     = delta.stride(1);
    p.out_batch_stride   = out.stride(0);
    p.out_d_stride       = out.stride(1);
    if (has_z) {
        p.z_batch_stride     = z.stride(0);
        p.z_d_stride         = z.stride(1);
        p.out_z_batch_stride = out_z.stride(0);
        p.out_z_d_stride     = out_z.stride(1);
    }
}

void set_ssm_params_bwd(SSMParamsBwd& p, const ScanConfig& cfg,
                        const at::Tensor& u, const at::Tensor& delta,
                        const at::Tensor& A, const at::Tensor& B, const at::Tensor& C,
                        const at::Tensor& z, const at::Tensor& out,
                        const at::Tensor& out_z,
                        void* D_ptr, void* delta_bias_ptr, void* x_ptr,
                        const at::Tensor& dout, const at::Tensor& du,
                        const at::Tensor& ddelta,
                        const at::Tensor& dA, const at::Tensor& dB, const at::Tensor& dC,
                        const at::Tensor& dz,
                        void* dD_ptr, void* ddelta_bias_ptr,
                        bool has_z, bool recompute_out_z)
{
    // Base fields. When the optional tensors are absent, dummy placeholders
    // (dout) are passed so the setter never touches empty Tensors.
    set_ssm_params_fwd(p, cfg,
                       u, delta, A, B, C,
                       has_z           ? out    : dout,
                       has_z           ? z      : dout,
                       recompute_out_z ? out_z  : dout,
                       D_ptr, delta_bias_ptr, x_ptr, has_z);
    if (!recompute_out_z) { p.out_z_ptr = nullptr; }

    // Gradient pointers
    p.dout_ptr        = dout.data_ptr();
    p.du_ptr          = du.data_ptr();
    p.ddelta_ptr      = ddelta.data_ptr();
    p.dA_ptr          = dA.data_ptr();
    p.dB_ptr          = dB.data_ptr();
    p.dC_ptr          = dC.data_ptr();
    p.dz_ptr          = has_z ? dz.data_ptr() : nullptr;
    p.dD_ptr          = dD_ptr;
    p.ddelta_bias_ptr = ddelta_bias_ptr;

    // Gradient strides
    p.dout_batch_stride  = dout.stride(0);
    p.dout_d_stride      = dout.stride(1);
    p.dA_d_stride        = dA.stride(0);
    p.dA_dstate_stride   = dA.stride(1);
    p.dB_dstate_stride   = cfg.is_variable_B ? dB.stride(2) : dB.stride(1);
    p.dC_dstate_stride   = cfg.is_variable_C ? dC.stride(2) : dC.stride(1);
    if (cfg.is_variable_B) {
        p.dB_batch_stride = dB.stride(0);
        p.dB_group_stride = dB.stride(1);
    } else {
        p.dB_d_stride = dB.stride(0);
    }
    if (cfg.is_variable_C) {
        p.dC_batch_stride = dC.stride(0);
        p.dC_group_stride = dC.stride(1);
    } else {
        p.dC_d_stride = dC.stride(0);
    }
    p.du_batch_stride     = du.stride(0);
    p.du_d_stride         = du.stride(1);
    p.ddelta_batch_stride = ddelta.stride(0);
    p.ddelta_d_stride     = ddelta.stride(1);
    if (has_z) {
        p.dz_batch_stride = dz.stride(0);
        p.dz_d_stride     = dz.stride(1);
    }
}

// ============================================================
// Public entry points (pybind-registered)
// ============================================================

std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor &u, const at::Tensor &delta,
                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                   const c10::optional<at::Tensor> &D_,
                   const c10::optional<at::Tensor> &z_,
                   const c10::optional<at::Tensor> &delta_bias_,
                   bool delta_softplus)
{
    const auto input_type  = u.scalar_type();
    const auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type  == at::ScalarType::Float ||
                input_type  == at::ScalarType::Half  ||
                input_type  == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float ||
                weight_type == at::ScalarType::ComplexFloat);

    const auto cfg = validate_scan_inputs(u, delta, A, B, C,
                                          input_type, weight_type, delta_softplus);

    if (D_.has_value()) {
        const auto& D = *D_;
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, cfg.dim);
    }
    if (delta_bias_.has_value()) {
        const auto& delta_bias = *delta_bias_;
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, cfg.dim);
    }

    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = *z_;
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, cfg.batch, cfg.dim, cfg.seqlen);
        out_z = torch::empty_like(z);
    }

    // u has BHL layout, delta has HBL layout; out mirrors delta (HBL).
    at::Tensor out = torch::empty_like(delta);
    at::Tensor x   = torch::empty({cfg.batch, cfg.dim, cfg.n_chunks, cfg.dstate * 2},
                                  u.options().dtype(weight_type));

    SSMParamsBase params{};
    set_ssm_params_fwd(params, cfg, u, delta, A, B, C, out, z, out_z,
                       D_.has_value()          ? D_->data_ptr()          : nullptr,
                       delta_bias_.has_value() ? delta_bias_->data_ptr() : nullptr,
                       x.data_ptr(), has_z);

    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(input_type, "selective_scan_fwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(weight_type, "selective_scan_fwd", [&] {
            selective_scan_fwd_cuda<input_t, weight_t>(params, stream);
        });
    });

    std::vector<at::Tensor> result = {out, x};
    if (has_z) { result.push_back(out_z); }
    return result;
}

std::vector<at::Tensor>
selective_scan_bwd(const at::Tensor &u, const at::Tensor &delta,
                   const at::Tensor &A, const at::Tensor &B, const at::Tensor &C,
                   const c10::optional<at::Tensor> &D_,
                   const c10::optional<at::Tensor> &z_,
                   const c10::optional<at::Tensor> &delta_bias_,
                   const at::Tensor &dout,
                   const c10::optional<at::Tensor> &x_,
                   const c10::optional<at::Tensor> &out_,
                   c10::optional<at::Tensor> &dz_,
                   bool delta_softplus,
                   bool recompute_out_z)
{
    const auto input_type  = u.scalar_type();
    const auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type  == at::ScalarType::Float ||
                input_type  == at::ScalarType::Half  ||
                input_type  == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float ||
                weight_type == at::ScalarType::ComplexFloat);

    const auto cfg = validate_scan_inputs(u, delta, A, B, C,
                                          input_type, weight_type, delta_softplus);

    // dout is bwd-specific; validate it alongside the shared tensors.
    TORCH_CHECK(dout.scalar_type() == input_type);
    TORCH_CHECK(dout.is_cuda());
    TORCH_CHECK(dout.stride(-1) == 1 || dout.size(-1) == 1);
    CHECK_SHAPE(dout, cfg.batch, cfg.dim, cfg.seqlen);

    if (D_.has_value()) {
        const auto& D = *D_;
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, cfg.dim);
    }
    if (delta_bias_.has_value()) {
        const auto& delta_bias = *delta_bias_;
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, cfg.dim);
    }

    at::Tensor z, out, dz, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = *z_;
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        CHECK_SHAPE(z, cfg.batch, cfg.dim, cfg.seqlen);

        TORCH_CHECK(out_.has_value());
        out = *out_;
        TORCH_CHECK(out.scalar_type() == input_type);
        TORCH_CHECK(out.is_cuda());
        TORCH_CHECK(out.stride(-1) == 1 || out.size(-1) == 1);
        CHECK_SHAPE(out, cfg.batch, cfg.dim, cfg.seqlen);

        if (dz_.has_value()) {
            dz = *dz_;
            TORCH_CHECK(dz.scalar_type() == input_type);
            TORCH_CHECK(dz.is_cuda());
            TORCH_CHECK(dz.stride(-1) == 1 || dz.size(-1) == 1);
            CHECK_SHAPE(dz, cfg.batch, cfg.dim, cfg.seqlen);
        } else {
            dz = torch::empty_like(z);
        }
        if (recompute_out_z) {
            out_z = torch::empty_like(out);
        }
    }

    if (cfg.n_chunks > 1) { TORCH_CHECK(x_.has_value()); }
    if (x_.has_value()) {
        const auto& x = *x_;
        TORCH_CHECK(x.scalar_type() == weight_type);
        TORCH_CHECK(x.is_cuda());
        TORCH_CHECK(x.is_contiguous());
        CHECK_SHAPE(x, cfg.batch, cfg.dim, cfg.n_chunks, 2 * cfg.dstate);
    }

    at::Tensor du     = torch::empty_like(u);
    at::Tensor ddelta = torch::empty_like(delta);
    at::Tensor dA     = torch::zeros_like(A);
    at::Tensor dB     = cfg.is_variable_B
        ? torch::zeros_like(B, B.options().dtype(torch::kFloat32))
        : torch::zeros_like(B);
    at::Tensor dC     = cfg.is_variable_C
        ? torch::zeros_like(C, C.options().dtype(torch::kFloat32))
        : torch::zeros_like(C);
    at::Tensor dD, ddelta_bias;
    if (D_.has_value())          { dD          = torch::zeros_like(*D_);          }
    if (delta_bias_.has_value()) { ddelta_bias = torch::zeros_like(*delta_bias_); }

    SSMParamsBwd params{};
    set_ssm_params_bwd(params, cfg, u, delta, A, B, C, z, out, out_z,
                       D_.has_value()          ? D_->data_ptr()          : nullptr,
                       delta_bias_.has_value() ? delta_bias_->data_ptr() : nullptr,
                       x_.has_value()          ? x_->data_ptr()          : nullptr,
                       dout, du, ddelta, dA, dB, dC, dz,
                       D_.has_value()          ? dD.data_ptr()          : nullptr,
                       delta_bias_.has_value() ? ddelta_bias.data_ptr() : nullptr,
                       has_z, recompute_out_z);

    at::cuda::CUDAGuard device_guard{u.device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(input_type, "selective_scan_bwd", [&] {
        DISPATCH_WTYPE_FLOAT_AND_COMPLEX(weight_type, "selective_scan_bwd", [&] {
            selective_scan_bwd_cuda<input_t, weight_t>(params, stream);
        });
    });

    std::vector<at::Tensor> result = {du, ddelta, dA, dB.to(B.dtype()), dC.to(C.dtype()), dD, ddelta_bias};
    if (has_z)           { result.push_back(dz);    }
    if (recompute_out_z) { result.push_back(out_z); }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &selective_scan_fwd, "Selective scan forward");
    m.def("bwd", &selective_scan_bwd, "Selective scan backward");
}