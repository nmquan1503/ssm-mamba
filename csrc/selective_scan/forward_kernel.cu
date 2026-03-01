#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_scan.cuh>

#include "kernel_config.h"
#include "common.h"

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