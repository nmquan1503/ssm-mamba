#include <cub/block/block_reduce.cuh>
#include <c10/cuda/CUDAException.h>

#include "selective_update.h"
#include "static_switch.h"

template<int kNumThreads_>
struct SUKernelTraits {
    static constexpr int kNumThreads = kNumThreads_;
    static constexpr int kMinBlocks = (kNumThreads < 128) ? 5 : 3;

    using BlockReduce = cub::BlockReduce<float, kNumThreads>;

    static constexpr int kSMemSizeInBytes = sizeof(typename BlockReduce::TempStorage);
};

template<typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads, Traits::kMinBlocks)
void kernel(SUParams& params) {
    extern __shared__ char smem_[];

    auto& smem_reduce = *reinterpret_cast<typename Traits::BlockReduce::TempStorage*>(smem_);

    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    const int state_id = threadIdx.x;

    float* u = reinterpret_cast<float*>(params.u_ptr);
    float* A = reinterpret_cast<float*>(params.A_ptr);
    float* B = reinterpret_cast<float*>(params.B_ptr);
    float* C = reinterpret_cast<float*>(params.C_ptr);
    float* D = reinterpret_cast<float*>(params.D_ptr);
    float* delta = reinterpret_cast<float*>(params.delta_ptr);
    float* delta_bias = reinterpret_cast<float*>(params.delta_bias_ptr);
    float* h = reinterpret_cast<float*>(params.h_ptr);
    float* new_h = reinterpret_cast<float*>(params.new_h_ptr);
    float* out = reinterpret_cast<float*>(params.out_ptr);

    float u_val = u[batch_id * params.u_batch_stride + channel_id];
    float A_val = state_id < params.state_dim 
        ? A[channel_id * params.A_channel_stride + state_id]
        : 0.f;
    float B_val = state_id < params.state_dim
        ? B[batch_id * params.B_batch_stride + state_id]
        : 0.f;
    float C_val = state_id < params.state_dim
        ? C[batch_id * params.C_batch_stride + state_id]
        : 0.f;
    float D_val = state_id == 0 ? D[channel_id] : 0.f;
    float delta_raw_val = delta[batch_id * params.delta_batch_stride + channel_id];
    float delta_bias_val = delta_bias[channel_id];
    float h_val = state_id < params.state_dim
        ? h[batch_id * params.h_batch_stride + channel_id * params.h_channel_stride + state_id]
        : 0.f;

    float delta_val = delta_raw_val + delta_bias_val;
    float new_h_val = state_id < params.state_dim
        ? exp2f(delta_val * A_val * M_LOG2E) * h_val + delta_val * u_val * B_val
        : 0.f;

    float out_val = C_val * new_h_val;
    out_val = typename Traits::BlockReduce(smem_reduce).Sum(out_val);
    
    if (state_id < params.state_dim) {
        new_h[batch_id * params.h_batch_stride + channel_id * params.h_channel_stride + state_id] = new_h_val;
    }
    if (state_id == 0) {
        out_val += D_val * u_val;
        out[batch_id * params.out_batch_stride + channel_id] = out_val;
    }
}

void kernel_launch(SUParams& params, cudaStream_t stream) {
    DISPATCH_SWITCH(params.state_dim, MAX_STATE_DIM, [&]{
        constexpr int kNumThreads = ((MAX_STATE_DIM - 1) / 32 + 1) * 32;
        using Traits = SUKernelTraits<kNumThreads>;
        
        constexpr int kSMemSizeInBytes = Traits::kSMemSizeInBytes;

        dim3 grid(params.batch_size, params.num_channels);
        auto ker = &kernel<Traits>;
        if (kSMemSizeInBytes >= 48 * 1024) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(
                ker, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMemSizeInBytes
            ));
        }
        ker<<<grid, Traits::kNumThreads, kSMemSizeInBytes, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}