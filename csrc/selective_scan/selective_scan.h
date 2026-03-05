#pragma once

#include <vector>

namespace at {
    class Tensor;
}

struct SSScanParams {
    int batch_size, seq_len, num_chunks;
    int A_batch_stride;
    int B_batch_stride;
    int out_batch_stride;

    void* __restrict__ A_ptr;
    void* __restrict__ B_ptr;
    void* __restrict__ h_ptr;
    void* __restrict__ out_ptr;
};

struct BaseSSParams {
    int batch_size, seq_len, state_dim, num_channels;

    int num_chunks;
    
    int u_batch_stride, u_channel_stride;
    int A_channel_stride;
    int B_batch_stride, B_state_stride;
    int C_batch_stride, C_state_stride;
    int delta_batch_stride, delta_channel_stride;

    void* __restrict__ u_ptr;   // (batch_size, num_channels, seq_len)
    void* __restrict__ A_ptr;   // (num_channels, state_dim)
    void* __restrict__ B_ptr;   // (batch_size, state_dim, seq_len)
    void* __restrict__ C_ptr;   // (batch_size, state_dim, seq_len)
    void* __restrict__ D_ptr;   // (num_channels,)
    void* __restrict__ delta_ptr;   // (batch_size, num_channels, seq_len)
    void* __restrict__ delta_bias_ptr;  // (num_channels,)
    void* __restrict__ h_ptr;   // (batch_size, num_channels, num_chunks, state_dim * 2)
};

struct ForwardSSParams : BaseSSParams {
    int out_batch_stride, out_channel_stride;
    int last_h_batch_stride, last_h_channel_stride;

    void* __restrict__ out_ptr; // (batch_size, num_channels, seq_len)
    void* __restrict__ length_ptr; // (batch_size,)
    void* __restrict__ last_h_ptr;  // (batch_size, num_channels, state_dim)
};

struct BackwardSSParams : BaseSSParams {
    int du_batch_stride, du_channel_stride;
    int dA_channel_stride;
    int dB_batch_stride, dB_state_stride;
    int dC_batch_stride, dC_state_stride;
    int ddelta_batch_stride, ddelta_channel_stride;
    int dout_batch_stride, dout_channel_stride;

    void* __restrict__ du_ptr;  // (batch_size, num_channels, seq_len)
    void* __restrict__ dA_ptr;  // (num_channels, state_dim)
    void* __restrict__ dB_ptr;  // (batch_size, state_dim, seq_len)
    void* __restrict__ dC_ptr;  // (batch_size, state_dim, seq_len)
    void* __restrict__ dD_ptr;  // (num_channels,)
    void* __restrict__ ddelta_ptr;  // (batch_size, num_channels, seq_len)
    void* __restrict__ ddelta_bias_ptr; // (num_channels,)
    void* __restrict__ dout_ptr;    // (batch_size, num_channels, seq_len)
};

std::vector<at::Tensor> selective_scan_forward(
    const at::Tensor& u,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& delta,
    const at::Tensor& delta_bias,
    const at::Tensor& length
);

std::vector<at::Tensor> selective_scan_backward(
    const at::Tensor& u,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& delta,
    const at::Tensor& delta_bias,
    const at::Tensor& h,
    const at::Tensor& dout
);