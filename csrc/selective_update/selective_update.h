#pragma once

#include <vector>

namespace at {
    class Tensor;
}

struct SUParams {
    int batch_size, state_dim, num_channels;

    int u_batch_stride;
    int A_channel_stride;
    int B_batch_stride;
    int C_batch_stride;
    int delta_batch_stride;
    int h_batch_stride, h_channel_stride;
    int out_batch_stride;

    void* __restrict__ u_ptr;   // (batch_size, num_channels)
    void* __restrict__ A_ptr;   // (num_channels, state_dim)
    void* __restrict__ B_ptr;   // (batch_size, state_dim)
    void* __restrict__ C_ptr;   // (batch_size, state_dim)
    void* __restrict__ D_ptr;   // (num_channels)
    void* __restrict__ delta_ptr;   // (batch_size, num_channels)
    void* __restrict__ delta_bias_ptr;  // (num_channels)
    void* __restrict__ h_ptr;   // (batch_size, num_channels, state_dim)
    void* __restrict__ new_h_ptr;   // (batch_size, num_channels, state_dim)
    void* __restrict__ out_ptr; // (batch_size, num_channels)
};

std::vector<at::Tensor> selective_update(
    const at::Tensor& u,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    const at::Tensor& D,
    const at::Tensor& delta,
    const at::Tensor& delta_bias,
    const at::Tensor& h
);