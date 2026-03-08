#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/ArrayRef.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <optional>

#include "selective_scan.h"
#include "kernel_config.h"

#define CHECK(x, ...)                                                   \
    do {                                                                \
        TORCH_CHECK((x).is_cuda(),                                      \
                    #x " must be a CUDA tensor");                       \
                                                                        \
        TORCH_CHECK((x).scalar_type() == at::kFloat,                    \
                    #x " must be float32 tensor");                      \
                                                                        \
        TORCH_CHECK((x).is_contiguous(),                                \
                    #x " must be contiguous");                          \
                                                                        \
        TORCH_CHECK((x).sizes() == c10::IntArrayRef({__VA_ARGS__}),     \
                    #x " must have shape (" #__VA_ARGS__ ")");          \
    } while (0)

#define CHECK_DIM(x, n)                                                 \
    TORCH_CHECK((x).dim() == (n),                                       \
                #x " must be a " #n "D tensor")

void forward_kernel_launch(ForwardSSParams& params, cudaStream_t stream);
void backward_kernel_launch(BackwardSSParams& params, cudaStream_t stream);

void set_base_params(
    BaseSSParams& params,
    const int batch_size,
    const int seq_len,
    const int state_dim,
    const int num_channels,
    const bool has_h_init,
    const int num_chunks,
    const at::Tensor u,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor C,
    void* D_ptr,
    const at::Tensor delta,
    void* delta_bias_ptr,
    const at::Tensor h,
    const at::Tensor h_init,
    void* length_ptr
) {
    memset(&params, 0, sizeof(params));

    params.batch_size = batch_size;
    params.seq_len = seq_len;
    params.state_dim = state_dim;
    params.num_channels = num_channels;
    params.has_h_init = has_h_init;
    params.num_chunks = num_chunks;

    params.u_ptr = u.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_ptr = delta.data_ptr();
    params.delta_bias_ptr = delta_bias_ptr;
    params.h_ptr = h.data_ptr();
    params.h_init_ptr = has_h_init ? h_init.data_ptr() : nullptr;
    params.length_ptr = length_ptr;

    params.u_batch_stride = u.stride(0);
    params.u_channel_stride = u.stride(1);
    
    params.A_channel_stride = A.stride(0);
    
    params.B_batch_stride = B.stride(0);
    params.B_state_stride = B.stride(1);

    params.C_batch_stride = C.stride(0);
    params.C_state_stride = C.stride(1);

    params.delta_batch_stride = delta.stride(0);
    params.delta_channel_stride = delta.stride(1);

    params.h_batch_stride = h.stride(0);
    params.h_channel_stride = h.stride(1);
    params.h_chunk_stride = h.stride(2);

    if (has_h_init) {
        params.h_init_batch_stride = h_init.stride(0);
        params.h_init_channel_stride = h_init.stride(1);
    }
}

void set_forward_params(
    ForwardSSParams& params,
    const int batch_size,
    const int seq_len,
    const int state_dim,
    const int num_channels,
    const bool has_h_init,
    const int num_chunks,
    const at::Tensor u,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor C,
    void* D_ptr,
    const at::Tensor delta,
    void* delta_bias_ptr,
    const at::Tensor h,
    const at::Tensor h_init,
    const at::Tensor out,
    void* length_ptr
) {
    set_base_params(
        params, 
        batch_size, seq_len, state_dim, num_channels, 
        has_h_init, num_chunks,
        u, A, B, C, D_ptr,
        delta, delta_bias_ptr, h, h_init, length_ptr
    );

    params.out_ptr = out.data_ptr();

    params.out_batch_stride = out.stride(0);
    params.out_channel_stride = out.stride(1);
}

void set_backward_params(
    BackwardSSParams& params,
    const int batch_size,
    const int seq_len,
    const int state_dim,
    const int num_channels,
    const bool has_h_init,
    const int num_chunks,
    const at::Tensor u,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor C,
    void* D_ptr,
    const at::Tensor delta,
    void* delta_bias_ptr,
    const at::Tensor h,
    const at::Tensor h_init,
    const at::Tensor du,
    const at::Tensor dA,
    const at::Tensor dB,
    const at::Tensor dC,
    void* dD_ptr,
    const at::Tensor ddelta,
    void* ddelta_bias_ptr,
    const at::Tensor dh_init,
    const at::Tensor dout,
    const at::Tensor dh_last,
    void* length_ptr
) {
    set_base_params(
        params, 
        batch_size, seq_len, state_dim, num_channels,
        has_h_init, num_chunks,
        u, A, B, C, D_ptr,
        delta, delta_bias_ptr, h, h_init, length_ptr
    );

    params.du_ptr = du.data_ptr();
    params.dA_ptr = dA.data_ptr();
    params.dB_ptr = dB.data_ptr();
    params.dC_ptr = dC.data_ptr();
    params.dD_ptr = dD_ptr;
    params.ddelta_ptr = ddelta.data_ptr();
    params.ddelta_bias_ptr = ddelta_bias_ptr;
    params.dh_init_ptr = has_h_init ? dh_init.data_ptr() : nullptr;
    params.dout_ptr = dout.data_ptr();
    params.dh_last_ptr = dh_last.data_ptr();

    params.du_batch_stride = du.stride(0);
    params.du_channel_stride = du.stride(1);
    
    params.dA_channel_stride = dA.stride(0);
    
    params.dB_batch_stride = dB.stride(0);
    params.dB_state_stride = dB.stride(1);

    params.dC_batch_stride = dC.stride(0);
    params.dC_state_stride = dC.stride(1);

    params.ddelta_batch_stride = ddelta.stride(0);
    params.ddelta_channel_stride = ddelta.stride(1);

    if (has_h_init) {
        params.dh_init_batch_stride = dh_init.stride(0);
        params.dh_init_channel_stride = dh_init.stride(1);
    }

    params.dout_batch_stride = dout.stride(0);
    params.dout_channel_stride = dout.stride(1);

    params.dh_last_batch_stride = dh_last.stride(0);
    params.dh_last_channel_stride = dh_last.stride(1);
}

std::vector<at::Tensor> selective_scan_forward(
    const at::Tensor& u,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& delta,
    const at::Tensor& delta_bias,
    const std::optional<at::Tensor>& h_init_,
    const std::optional<at::Tensor>& length_
) {
    const bool has_h_init = h_init_.has_value();
    const bool has_length = length_.has_value();
    const at::Tensor h_init = has_h_init ? h_init_.value() : at::Tensor();
    const at::Tensor length = has_length ? length_.value() : at::Tensor();

    CHECK_DIM(u, 3);
    CHECK_DIM(A, 2);
    CHECK_DIM(B, 3);
    CHECK_DIM(C, 3);
    CHECK_DIM(D, 1);
    CHECK_DIM(delta, 3);
    CHECK_DIM(delta_bias, 1);
    if (has_h_init) {
        CHECK_DIM(h_init, 3);
    }
    if (has_length) {
        CHECK_DIM(length, 1);
    }

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int num_channels = sizes[1];
    const int seq_len = sizes[2];
    const int state_dim = A.size(1);
    const int num_chunks = kernel_config::get_num_chunks(seq_len);

    TORCH_CHECK(state_dim <= 256, "selective_scan only supports state_dim <= 256");

    CHECK(u, batch_size, num_channels, seq_len);
    CHECK(A, num_channels, state_dim);
    CHECK(B, batch_size, state_dim, seq_len);
    CHECK(C, batch_size, state_dim, seq_len);
    CHECK(D, num_channels);
    CHECK(delta, batch_size, num_channels, seq_len);
    CHECK(delta_bias, num_channels);
    if (has_h_init) {
        CHECK(h_init, batch_size, num_channels, state_dim);
    }
    if  (has_length) {
        TORCH_CHECK(length.is_cuda(),"length must be a CUDA tensor");
        TORCH_CHECK(length.scalar_type() == at::kLong, "length must be long tensor");
        TORCH_CHECK(length.is_contiguous(),"length must be contiguous");
        TORCH_CHECK(length.size(0) == batch_size, "length must have shape (batch_size,)");
    }

    at::Tensor out = at::empty_like(u);
    at::Tensor h = at::empty(
        {batch_size, num_channels, num_chunks, state_dim * 2},
        u.options()
    );

    ForwardSSParams params;
    set_forward_params(
        params,
        batch_size, seq_len, state_dim, num_channels, 
        has_h_init, num_chunks,
        u, A, B, C, D.data_ptr(),
        delta, delta_bias.data_ptr(),
        h, h_init, out, has_length ? length.data_ptr() : nullptr
    );

    at::cuda::CUDAGuard device_guard(u.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    forward_kernel_launch(params, stream);
    std::vector<at::Tensor> result = {out, h};
    
    return result;
}

std::vector<at::Tensor> selective_scan_backward(
    const at::Tensor& u,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& delta,
    const at::Tensor& delta_bias,
    const at::Tensor& h,
    const std::optional<at::Tensor>& h_init_,
    const at::Tensor& dout,
    const at::Tensor& dh_last,
    const std::optional<at::Tensor>& length_
) {
    const bool has_h_init = h_init_.has_value();
    const bool has_length = length_.has_value();
    const at::Tensor h_init = has_h_init ? h_init_.value() : at::Tensor();
    const at::Tensor length = has_length ? length_.value() : at::Tensor();

    CHECK_DIM(u, 3);
    CHECK_DIM(A, 2);
    CHECK_DIM(B, 3);
    CHECK_DIM(C, 3);
    CHECK_DIM(D, 1);
    CHECK_DIM(delta, 3);
    CHECK_DIM(delta_bias, 1);
    CHECK_DIM(h, 4);
    if (has_h_init) {
        CHECK_DIM(h_init, 3);
    }
    CHECK_DIM(dout, 3);
    if (has_length) {
        CHECK_DIM(length, 1);
    }

    const auto sizes = u.sizes();
    const int batch_size = sizes[0];
    const int num_channels = sizes[1];
    const int seq_len = sizes[2];
    const int state_dim = A.size(1);
    const int num_chunks = kernel_config::get_num_chunks(seq_len);

    TORCH_CHECK(state_dim <= 256, "selective_scan only supports state_dim <= 256");

    CHECK(u, batch_size, num_channels, seq_len);
    CHECK(A, num_channels, state_dim);
    CHECK(B, batch_size, state_dim, seq_len);
    CHECK(C, batch_size, state_dim, seq_len);
    CHECK(D, num_channels);
    CHECK(delta, batch_size, num_channels, seq_len);
    CHECK(delta_bias, num_channels);
    CHECK(h, batch_size, num_channels, num_chunks, state_dim * 2);
    if (has_h_init) {
        CHECK(h_init, batch_size, num_channels, state_dim);
    }
    CHECK(dout, batch_size, num_channels, seq_len);

    at::Tensor du = at::empty_like(u);
    at::Tensor dA = at::zeros_like(A);
    at::Tensor dB = at::zeros_like(B);
    at::Tensor dC = at::zeros_like(C);
    at::Tensor dD = at::zeros_like(D);
    at::Tensor ddelta = at::empty_like(delta);
    at::Tensor ddelta_bias = at::zeros_like(delta_bias);
    at::Tensor dh_init = has_h_init ? at::empty_like(h_init) : at::Tensor();

    BackwardSSParams params;
    set_backward_params(
        params,
        batch_size, seq_len, state_dim, num_channels, 
        has_h_init, num_chunks,
        u, A, B, C, D.data_ptr(),
        delta, delta_bias.data_ptr(), h, h_init,
        du, dA, dB, dC, dD.data_ptr(),
        ddelta, ddelta_bias.data_ptr(), dh_init, 
        dout, dh_last, has_length ? length.data_ptr() : nullptr
    );

    at::cuda::CUDAGuard device_guard(u.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    backward_kernel_launch(params, stream);
    std::vector<at::Tensor> result = {du, dA, dB, dC, dD, ddelta, ddelta_bias, dh_init};

    return result;
}