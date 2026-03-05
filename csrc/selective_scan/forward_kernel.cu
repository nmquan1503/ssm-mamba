#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <c10/cuda/CUDAException.h>

#include "kernel_config.h"
#include "common.h"
#include "selective_scan.h"
#include "static_switch.h"

template<bool kSeqDivisible_>
struct SSForwardKernelTraits {
    static constexpr int kNumThreads = kernel_config::num_threads;
    static constexpr int kMinBlocks = (kNumThreads < 128) ? 5 : 3;
    static constexpr int kNumElements = kernel_config::num_elements;
    static constexpr int kNumVectors = kNumElements / 4;
    static constexpr bool kSeqDivisible = kSeqDivisible_;
    static constexpr bool kEnableDirectVectorIO = kSeqDivisible && (kNumVectors == 1);

    using ScalarBlockLoad = cub::BlockLoad<
        float, 
        kNumThreads, 
        kNumElements,
        cub::BLOCK_LOAD_WARP_TRANSPOSE
    >;

    using VectorBlockLoad = cub::BlockLoad<
        float4,
        kNumThreads,
        kNumVectors,
        kEnableDirectVectorIO
            ? cub::BLOCK_LOAD_DIRECT
            : cub::BLOCK_LOAD_WARP_TRANSPOSE
    >;

    using ScalarBlockStore = cub::BlockStore<
        float,
        kNumThreads,
        kNumElements,
        cub::BLOCK_STORE_WARP_TRANSPOSE
    >;

    using VectorBlockStore = cub::BlockStore<
        float4,
        kNumThreads,
        kNumVectors,
        kEnableDirectVectorIO
            ? cub::BLOCK_STORE_DIRECT
            : cub::BLOCK_STORE_WARP_TRANSPOSE
    >;

    using BlockScan = cub::BlockScan<
        float2,
        kNumThreads,
        cub::BLOCK_SCAN_WARP_SCANS
    >;

    static constexpr int kSMemIOSizeInBytes = max_of({
        sizeof(typename ScalarBlockLoad::TempStorage) * 2, // for both B and C
        sizeof(typename VectorBlockLoad::TempStorage) * 2, // for both B and C
        sizeof(typename ScalarBlockStore::TempStorage),
        sizeof(typename VectorBlockStore::TempStorage)
    });

    static constexpr int kSMemSizeInBytes = kSMemIOSizeInBytes + sizeof(typename BlockScan::TempStorage);
};

template<typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads, Traits::kMinBlocks)
void forward_kernel(ForwardSSParams params) {
    constexpr int kNumThreads = Traits::kNumThreads;
    constexpr int kNumElements = Traits::kNumElements;
    constexpr int kEnableDirectVectorIO = Traits::kEnableDirectVectorIO;

    extern __shared__ char smem_[];

    auto& smem_load_primary = reinterpret_cast<
        typename Traits::ScalarBlockLoad::TempStorage&
    >(smem_);

    auto& smem_load_secondary = *reinterpret_cast<
        typename Traits::ScalarBlockLoad::TempStorage*
    >(smem_ + sizeof(typename Traits::ScalarBlockLoad::TempStorage));

    auto& smem_store = reinterpret_cast<
        typename Traits::ScalarBlockStore::TempStorage&
    >(smem_);

    auto& smem_scan = *reinterpret_cast<
        typename Traits::BlockScan::TempStorage*
    >(smem_ + Traits::kSMemIOSizeInBytes);

    float2* smem_running_prefix = reinterpret_cast<float2*>(smem_ + Traits::kSMemSizeInBytes);

    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    
    float* u = reinterpret_cast<float*>(params.u_ptr)
        + batch_id * params.u_batch_stride
        + channel_id * params.u_channel_stride;

    float* A = reinterpret_cast<float*>(params.A_ptr)
        + channel_id * params.A_channel_stride;

    float* B = reinterpret_cast<float*>(params.B_ptr)
        + batch_id * params.B_batch_stride;
    
    float* C = reinterpret_cast<float*>(params.C_ptr)
        + batch_id * params.C_batch_stride;
    
    float D_val = reinterpret_cast<float*>(params.D_ptr)[channel_id];

    float* delta = reinterpret_cast<float*>(params.delta_ptr)
        + batch_id * params.delta_batch_stride
        + channel_id * params.delta_channel_stride;
    
    float delta_bias_val = reinterpret_cast<float*>(params.delta_bias_ptr)[channel_id];

    float2* h = reinterpret_cast<float2*>(params.h_ptr)
        + (batch_id * params.num_channels + channel_id) * params.num_chunks * params.state_dim;

    float* out = reinterpret_cast<float*>(params.out_ptr)
        + batch_id * params.out_batch_stride
        + channel_id * params.out_channel_stride;

    constexpr int kChunkSize = kernel_config::chunk_size;

    for (int chunk_id = 0; chunk_id < params.num_chunks; chunk_id++) {
        float u_vals[kNumElements], delta_raw_vals[kNumElements];
        __syncthreads();
        load<Traits>(u, u_vals, smem_load_primary, params.seq_len - chunk_id * kChunkSize);
        if constexpr (!kEnableDirectVectorIO) {
            // Needed when using shared memory staging
            __syncthreads();
        }
        load<Traits>(delta, delta_raw_vals, smem_load_primary, params.seq_len - chunk_id * kChunkSize);
        
        u += kChunkSize;
        delta += kChunkSize;

        // delta_t  = softplus(delta_raw_t + delta_bias)
        // B_t base term = delta_t * u_t
        // y_t init = D_s * u_t
        float delta_vals[kNumElements], delta_u_vals[kNumElements], out_vals[kNumElements];
        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            delta_vals[i] = delta_raw_vals[i] + delta_bias_val;
            // softplus
            delta_vals[i] = delta_vals[i] <= 20.f
                ? log1pf(expf(delta_vals[i]))
                : delta_vals[i];
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
            out_vals[i] = D_val * u_vals[i];
        }

        __syncthreads();
        
        for (int state_id = 0; state_id < params.state_dim; state_id++) {
            //  exp(x) = 2 ^ (x * log2(e))
            float A_val = A[state_id] * M_LOG2E;
            float B_vals[kNumElements], C_vals[kNumElements];
            load<Traits>(
                B + state_id * params.B_state_stride, 
                B_vals, 
                smem_load_primary, 
                params.seq_len - chunk_id * kChunkSize
            );
            load<Traits>(
                C + state_id * params.C_state_stride, 
                C_vals, 
                smem_load_secondary, 
                params.seq_len - chunk_id * kChunkSize
            );

            // A_t = exp(A_s * delta_t)
            // B_t = B_{s,t} * (delta_t * u_t)
            //
            // Recurrence:
            // h_t = A_t * h_{t-1} + B_t
            float2 h_vals[kNumElements];

            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                h_vals[i] = make_float2(
                    exp2f(delta_vals[i] * A_val),
                    delta_u_vals[i] * B_vals[i]
                );
                if constexpr (!Traits::kSeqDivisible) {
                    if (threadIdx.x * kNumElements + i >= params.seq_len - chunk_id * kChunkSize) {
                        h_vals[i] = make_float2(1.f, 0.f);
                    }
                }
            }

            float2 running_prefix;
            running_prefix = chunk_id > 0 && threadIdx.x % 32 == 0
                ? smem_running_prefix[state_id]
                : make_float2(1.f, 0.f);
            
            StateCarryCallbackOp prefix_callback(running_prefix);
            // Inclusive scan with operator:
            // (a1,b1) ⊗ (a0,b0) = (a1 * a0, a1 * b0 + b1)
            typename Traits::BlockScan(smem_scan).InclusiveScan(
                h_vals, h_vals, ScanOp(), prefix_callback
            );

            // Save final state of current chunk
            if (threadIdx.x == 0) {
                smem_running_prefix[state_id] = prefix_callback.carry;
                h[chunk_id * params.state_dim + state_id] = prefix_callback.carry;
            }

            // y_t += C_{s,t} * h_t
            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                out_vals[i] += h_vals[i].y * C_vals[i];
            }
        }
        
        __syncthreads();
        store<Traits>(out_vals, out, smem_store, params.seq_len - chunk_id * kChunkSize);
        
        B += kChunkSize;
        C += kChunkSize;
        out += kChunkSize;
    }
}

void forward_kernel_launch(ForwardSSParams& params, cudaStream_t stream) {
    BOOL_SWITCH(params.seq_len % kernel_config::chunk_size == 0, kSeqDivisible, [&]{
        using Traits = SSForwardKernelTraits<kSeqDivisible>;
        DISPATCH_SWITCH(params.state_dim, MAX_STATE_DIM, [&]{
            constexpr int kSMemSizeInBytes = Traits::kSMemSizeInBytes + MAX_STATE_DIM * sizeof(float2);
            dim3 grid(params.batch_size, params.num_channels);
            auto kernel = &forward_kernel<Traits>;
            if (kSMemSizeInBytes >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMemSizeInBytes
                ));
            }
            kernel<<<grid, Traits::kNumThreads, kSMemSizeInBytes, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}