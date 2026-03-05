#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/Atomic.cuh>

#include "kernel_config.h"
#include "reverse_scan.cuh"
#include "selective_scan.h"
#include "common.h"
#include "static_switch.h"

template<bool kSeqDivisible_, int kMaxStateDim_>
struct BackwardSSKernelTraits {
    static constexpr int kNumThreads = kernel_config::num_threads;
    static constexpr int kMinBlocks = (kNumThreads < 128) ? 5 : 3;
    static constexpr int kNumElements = kernel_config::num_elements;
    static constexpr int kNumVectors = kNumElements / 4;
    static constexpr bool kSeqDivisible = kSeqDivisible_;
    static constexpr int kMaxStateDim = kMaxStateDim_;
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
        cub::BLOCK_SCAN_RAKING
    >;

    using BlockReverseScan = BlockReverseScan<float2, kNumThreads>;

    using ScalarBlockReduce = cub::BlockReduce<float, kNumThreads>;

    using StateBlockReduce = cub::BlockReduce<float2, kNumThreads>;

    using BlockExchange = cub::BlockExchange<float, kNumThreads, kNumElements>;

    static constexpr int kSMemIOSizeInBytes = max_of({
        sizeof(typename ScalarBlockLoad::TempStorage),
        sizeof(typename VectorBlockLoad::TempStorage),
        sizeof(typename ScalarBlockStore::TempStorage),
        sizeof(typename VectorBlockStore::TempStorage)
    });

    static constexpr int kSMemExchangeSizeInBytes = sizeof(typename BlockExchange::TempStorage);

    static constexpr int kSMemReduceSizeInBytes = sizeof(typename StateBlockReduce::TempStorage);

    static constexpr int kSMemSizeInBytes = kSMemIOSizeInBytes
        + kSMemExchangeSizeInBytes
        + kSMemReduceSizeInBytes
        + sizeof(typename BlockScan::TempStorage) + sizeof(typename BlockReverseScan::TempStorage);
};

template<typename Traits>
__global__ __launch_bounds__(Traits::kNumThreads, Traits::kMinBlocks)
void backward_kernel(BackwardSSParams params) {
    constexpr int kNumThreads = Traits::kNumThreads;
    constexpr int kNumElements = Traits::kNumElements;
    constexpr int kMaxStateDim = Traits::kMaxStateDim;
    constexpr int kChunkSize = kernel_config::chunk_size;

    extern __shared__ char smem_[];

    auto& smem_load = reinterpret_cast<
        typename Traits::ScalarBlockLoad::TempStorage&
    >(smem_);

    auto& smem_store = reinterpret_cast<
        typename Traits::ScalarBlockStore::TempStorage&
    >(smem_);

    auto& smem_exchange = *reinterpret_cast<
        typename Traits::BlockExchange::TempStorage*
    >(smem_ + Traits::kSMemIOSizeInBytes);

    auto& smem_reduce = *reinterpret_cast<
        typename Traits::ScalarBlockReduce::TempStorage*
    >(reinterpret_cast<char*>(&smem_exchange) + Traits::kSMemExchangeSizeInBytes);

    auto& smem_scan = *reinterpret_cast<
        typename Traits::BlockScan::TempStorage*
    >(reinterpret_cast<char*>(&smem_reduce) + Traits::kSMemReduceSizeInBytes);

    auto& smem_reverse_scan = *reinterpret_cast<
        typename Traits::BlockReverseScan::TempStorage*
    >(
        reinterpret_cast<char*>(&smem_scan) + sizeof(typename Traits::BlockScan::TempStorage)
    );

    float* smem_delta_a = reinterpret_cast<float*>(smem_ + Traits::kSMemSizeInBytes);
    float2* smem_running_postfix = reinterpret_cast<float2*>(smem_delta_a + 2 * kMaxStateDim + kNumThreads);
    float* smem_da = reinterpret_cast<float*>(smem_running_postfix + kMaxStateDim);

    const int batch_id = blockIdx.x;
    const int channel_id = blockIdx.y;
    
    float* u = reinterpret_cast<float*>(params.u_ptr)
        + batch_id * params.u_batch_stride
        + channel_id * params.u_channel_stride
        + (params.num_chunks - 1) * kChunkSize;

    float* A = reinterpret_cast<float*>(params.A_ptr)
        + channel_id * params.A_channel_stride;

    float* B = reinterpret_cast<float*>(params.B_ptr)
        + batch_id * params.B_batch_stride
        + (params.num_chunks - 1) * kChunkSize;
    
    float* C = reinterpret_cast<float*>(params.C_ptr)
        + batch_id * params.C_batch_stride
        + (params.num_chunks - 1) * kChunkSize;
    
    float D_val = reinterpret_cast<float*>(params.D_ptr)[channel_id];

    float* delta = reinterpret_cast<float*>(params.delta_ptr)
        + batch_id * params.delta_batch_stride
        + channel_id * params.delta_channel_stride
        + (params.num_chunks - 1) * kChunkSize;
    
    float delta_bias_val = reinterpret_cast<float*>(params.delta_bias_ptr)[channel_id];

    float2* h = reinterpret_cast<float2*>(params.h_ptr)
        + (batch_id * params.num_channels + channel_id) * params.num_chunks * params.state_dim;

    float* du = reinterpret_cast<float*>(params.du_ptr) 
        + batch_id * params.du_batch_stride
        + channel_id * params.du_channel_stride
        + (params.num_chunks - 1) * kChunkSize;

    float* dA = reinterpret_cast<float*>(params.dA_ptr)
        + channel_id * params.dA_channel_stride;

    float* dB = reinterpret_cast<float*>(params.dB_ptr)
        + batch_id * params.dB_batch_stride
        + (params.num_chunks - 1) * kChunkSize;
        
    float* dC = reinterpret_cast<float*>(params.dC_ptr)
        + batch_id * params.dC_batch_stride
        + (params.num_chunks - 1) * kChunkSize;

    float* dD = reinterpret_cast<float*>(params.dD_ptr) + channel_id;
    float dD_val = 0.f;

    float* ddelta = reinterpret_cast<float*>(params.ddelta_ptr)
        + batch_id * params.ddelta_batch_stride
        + channel_id * params.ddelta_channel_stride
        + (params.num_chunks - 1) * kChunkSize;

    float* ddelta_bias = reinterpret_cast<float*>(params.ddelta_bias_ptr) + channel_id;
    float ddelta_bias_val = 0.f;

    float* dout = reinterpret_cast<float*>(params.dout_ptr)
        + batch_id * params.dout_batch_stride
        + channel_id * params.dout_channel_stride
        + (params.num_chunks - 1) * kChunkSize;
    
    for (int chunk_id = params.num_chunks - 1; chunk_id >= 0; chunk_id--) {
        float u_vals[kNumElements];
        float delta_raw_vals[kNumElements];
        float delta_vals[kNumElements];
        float delta_u_vals[kNumElements];

        float dout_vals[kNumElements];
        float du_vals[kNumElements];
        float ddelta_vals[kNumElements] = {0};

        __syncthreads();
        load<Traits>(u, u_vals, smem_load, params.seq_len - chunk_id * kChunkSize);
        __syncthreads();
        load<Traits>(delta, delta_raw_vals, smem_load, params.seq_len - chunk_id * kChunkSize);
        __syncthreads();
        load<Traits>(dout, dout_vals, smem_load, params.seq_len - chunk_id * kChunkSize);
        
        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            delta_vals[i] = delta_raw_vals[i] + delta_bias_val;
            delta_vals[i] = delta_vals[i] <= 20.f
                ? log1pf(expf(delta_vals[i]))
                : delta_vals[i];
            
            delta_u_vals[i] = delta_vals[i] * u_vals[i];
        }
        
        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            du_vals[i] = D_val * dout_vals[i];
        }

        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            dD_val += dout_vals[i] * u_vals[i];
        }

        __syncthreads();
        for (int state_id = 0; state_id < params.state_dim; state_id++) {
            float A_val = A[state_id];
            float B_vals[kNumElements];
            float C_vals[kNumElements];
            float dA_val = 0.f;
            float dB_vals[kNumElements];
            float dC_vals[kNumElements];

            load<Traits>(
                B + state_id * params.B_state_stride,
                B_vals,
                smem_load,
                params.seq_len - chunk_id * kChunkSize
            );
            __syncthreads();
            load<Traits>(
                C + state_id * params.C_state_stride,
                C_vals,
                smem_load,
                params.seq_len - chunk_id * kChunkSize
            );

            float2 h_vals[kNumElements];
            float2 dh_vals[kNumElements];
            
            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                const float delta_a_exp = exp2f(delta_vals[i] * A_val * M_LOG2E);
                h_vals[i] = make_float2(
                    delta_a_exp,
                    delta_u_vals[i] * B_vals[i]
                );

                if (i == 0) {
                    smem_delta_a[
                        threadIdx.x == 0
                            ? state_id + (chunk_id % 2) * kMaxStateDim
                            : threadIdx.x + 2 * kMaxStateDim
                    ] = delta_a_exp;
                }
                else {
                    dh_vals[i - 1].x = delta_a_exp;
                }
                dh_vals[i].y = dout_vals[i] * C_vals[i];
            }

            __syncthreads();
            dh_vals[kNumElements - 1].x = threadIdx.x == kNumThreads - 1
                ? (chunk_id == params.num_chunks - 1 ? 1.f : smem_delta_a[state_id + ((chunk_id + 1) % 2) * kMaxStateDim])
                : smem_delta_a[threadIdx.x + 1 + 2 * kMaxStateDim];
            
            float2 running_prefix = chunk_id > 0 && threadIdx.x % 32 == 0
                ? h[(chunk_id - 1) * params.state_dim + state_id]
                : make_float2(1.f, 0.f);
            StateCarryCallbackOp prefix_op(running_prefix);
            typename Traits::BlockScan(smem_scan).InclusiveScan(h_vals, h_vals, ScanOp(), prefix_op);

            float2 running_postfix = chunk_id < params.num_chunks - 1 && threadIdx.x % 32 == 0
                ? smem_running_postfix[state_id]
                : make_float2(1.f, 0.f);
            StateCarryCallbackOp postfix_op(running_postfix);
            typename Traits::BlockReverseScan(smem_reverse_scan).InclusiveReverseScan(dh_vals, dh_vals, ScanOp(), postfix_op);
            if (threadIdx.x == 0) {
                smem_running_postfix[state_id] = postfix_op.carry;
            }

            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                const float dh_val = dh_vals[i].y;
                const float ddelta_u = dh_val * B_vals[i];
                du_vals[i] += ddelta_u * delta_vals[i];
                const float at_ht = h_vals[i].y - (delta_u_vals[i] * B_vals[i]);
                ddelta_vals[i] += ddelta_u * u_vals[i] + dh_val * A_val * at_ht;
                dA_val += dh_val * delta_vals[i] * at_ht;
                dB_vals[i] = dh_val * delta_u_vals[i];
                dC_vals[i] = dout_vals[i] * h_vals[i].y;
            }

            typename Traits::BlockExchange(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
            __syncthreads();
            typename Traits::BlockExchange(smem_exchange).BlockedToStriped(dC_vals, dC_vals);

            const int seqlen_remaining = params.seq_len - chunk_id * kChunkSize - threadIdx.x;

            float* dB_cur = dB + state_id * params.dB_state_stride + threadIdx.x;
            float* dC_cur = dC + state_id * params.dC_state_stride + threadIdx.x;

            #pragma unroll
            for (int i = 0; i < kNumElements; i++) {
                if (i * kNumThreads < seqlen_remaining) {
                    gpuAtomicAdd(dB_cur + i * kNumThreads, dB_vals[i]);
                    gpuAtomicAdd(dC_cur + i * kNumThreads, dC_vals[i]);
                }
            }

            dA_val = typename Traits::ScalarBlockReduce(smem_reduce).Sum(dA_val);

            if (threadIdx.x == 0) {
                smem_da[state_id] = chunk_id == params.num_chunks - 1
                    ? dA_val
                    : dA_val + smem_da[state_id];
            }
        }

        #pragma unroll
        for (int i = 0; i < kNumElements; i++) {
            float delta_val = delta_raw_vals[i] + delta_bias_val;
            ddelta_vals[i] = delta_val <= 20.f
                ? ddelta_vals[i] / (1.f + expf(-delta_val))
                : ddelta_vals[i];
            ddelta_bias_val += ddelta_vals[i];
        }
        
        __syncthreads();
        store<Traits>(du_vals, du, smem_store, params.seq_len - chunk_id * kChunkSize);
        
        __syncthreads();
        store<Traits>(ddelta_vals, ddelta, smem_store, params.seq_len - chunk_id * kChunkSize);

        u -= kChunkSize;
        B -= kChunkSize;
        C -= kChunkSize;
        delta -= kChunkSize;

        du -= kChunkSize;
        dB -= kChunkSize;
        dC -= kChunkSize;
        ddelta -= kChunkSize;
        dout -= kChunkSize;
    }

    dD_val = typename Traits::ScalarBlockReduce(smem_reduce).Sum(dD_val);
    if (threadIdx.x == 0) {
        gpuAtomicAdd(dD, dD_val);
    }

    __syncthreads();
    ddelta_bias_val = typename Traits::ScalarBlockReduce(smem_reduce).Sum(ddelta_bias_val);
    if (threadIdx.x == 0) {
        gpuAtomicAdd(ddelta_bias, ddelta_bias_val);
    }

    for (int state_id = threadIdx.x; state_id < params.state_dim; state_id += blockDim.x) {
        gpuAtomicAdd(&(dA[state_id]), smem_da[state_id]);
    }
}


void backward_kernel_launch(BackwardSSParams& params, cudaStream_t stream) {
    BOOL_SWITCH(params.seq_len % kernel_config::chunk_size == 0, kSeqDivisible, [&]{
        DISPATCH_SWITCH(params.state_dim, MAX_STATE_DIM, [&]{
            using Traits = BackwardSSKernelTraits<kSeqDivisible, MAX_STATE_DIM>;

            constexpr int kSMemSizeInBytes = Traits::kSMemSizeInBytes 
                + MAX_STATE_DIM * sizeof(float2)
                + (kernel_config::num_threads + 3 * MAX_STATE_DIM) * sizeof(float);
            
            dim3 grid(params.batch_size, params.num_channels);
            auto kernel = &backward_kernel<Traits>;
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