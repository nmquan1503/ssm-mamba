#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <c10/cuda/CUDAException.h>

#include "kernel_config.h"
#include "reverse_scan.cuh"
#include "selective_scan.h"
#include "common.h"

template<bool kSeqDevisible_>
struct BackwardSSKernelTraits {
    static constexpr int kNumThreads = kernel_config::num_threads;
    static constexpr int kMinBlocks = (kNumThreads < 128) ? 5 : 3;
    static constexpr int kNumElements = kernel_config::num_elements;
    static constexpr int kNumVectors = kNumElements / 4;
    static constexpr bool kSeqDivisible = kSeqDivisible_;
    static constexpr bool kEnableDirectVectorIO = kSeqDivisible && (kNumVectors == 1);

    using ScalarBlockLoad = cub::BlockLoad<
        float, 
        kNumThreads, 
        kMinBlocks,
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
        sizeof(typename ScalarBlockLoad::TempStorage) * 2,
        sizeof(typename VectorBlockLoad::TempStorage) * 2,
        sizeof(typename ScalarBlockLoad::TempStorage),
        sizeof(typename VectorBlockLoad::TempStorage)
    });

    static constexpr int kSMemExchangeSizeInBytes = sizeof(typename BlockExchange::TempStorage) * 2;   // for both B and C

    static constexpr int kSMemReduceSizeInBytes = sizeof(typename StateBlockReduce::TempStorage);

    static constexpr int kSMemSizeInBytes = kSMemIOSizeInBytes
        + kSMemExchangeSizeInBytes
        + kSMemReduceSizeInBytes
        + sizeof(typename BlockScan::TempStorage) + sizeof(typename BlockReverseScan::TempStorage);
};
