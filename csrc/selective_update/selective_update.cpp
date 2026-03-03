#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>
#include <c10/util/ArrayRef.h>
#include <ATen/ops/empty_like.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#include "selective_update.h"

#define CHECK(x, ...)                                                   \
    do {                                                                \
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

void kernel_launch(SUParams& params, cudaStream_t stream);

void set_params(
    SUParams& params,
    const int batch_size,
    const int state_dim,
    const int num_channels,
    const at::Tensor u,
    const at::Tensor A,
    const at::Tensor B,
    const at::Tensor C,
    void* D_ptr,
    const at::Tensor delta,
    void* delta_bias_ptr,
    const at::Tensor h,
    void* new_h_ptr,
    const at::Tensor out
) {
    memset(&params, 0, sizeof(params));

    params.batch_size = batch_size;
    params.state_dim = state_dim;
    params.num_channels = num_channels;

    params.u_ptr = u.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D_ptr;
    params.delta_ptr = delta.data_ptr();
    params.delta_bias_ptr = delta_bias_ptr;
    params.h_ptr = h.data_ptr();
    params.new_h_ptr = new_h_ptr;
    params.out_ptr = out.data_ptr();

    params.u_batch_stride = u.stride(0);

    params.A_channel_stride = A.stride(0);

    params.B_batch_stride = B.stride(0);

    params.C_batch_stride = C.stride(0);

    params.delta_batch_stride = delta.stride(0);

    params.h_batch_stride = h.stride(0);
    params.h_channel_stride = h.stride(1);

    params.out_batch_stride = out.stride(0);
}

std::vector<at::Tensor> selective_update(
    const at::Tensor& u,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& delta,
    const at::Tensor& delta_bias,
    const at::Tensor& h
) {
    CHECK_DIM(u, 2);
    CHECK_DIM(A, 2);
    CHECK_DIM(B, 2);
    CHECK_DIM(C, 2);
    CHECK_DIM(D, 1);
    CHECK_DIM(delta, 2);
    CHECK_DIM(delta_bias, 1);
    CHECK_DIM(h, 3);

    const auto sizes = h.sizes();
    const int batch_size = sizes[0];
    const int num_channels = sizes[1];
    const int state_dim = sizes[2];

    TORCH_CHECK(state_dim <= 256, "selective_update only supports state_dim <= 256");

    CHECK(u, batch_size, num_channels);    
    CHECK(A, num_channels, state_dim);
    CHECK(B, batch_size, state_dim);
    CHECK(C, batch_size, state_dim);
    CHECK(D, num_channels);
    CHECK(delta, batch_size, num_channels);
    CHECK(h, batch_size, num_channels, state_dim);

    at::Tensor out = at::empty_like(u);
    at::Tensor new_h = at::empty_like(h);

    SUParams params;
    set_params(
        params,
        batch_size, state_dim, num_channels,
        u, A, B, C, D.data_ptr(),
        delta, delta_bias.data_ptr(),
        h, new_h.data_ptr(), out
    );

    at::cuda::CUDAGuard device_guard(u.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    kernel_launch(params, stream);
    std::vector<at::Tensor> result = {out, new_h};
    
    return result;
}