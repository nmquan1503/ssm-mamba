#pragma once

#include <cub/config.cuh>
#include <cub/util_type.cuh>
#include <cub/util_ptx.cuh>
#include <cub/block/block_raking_layout.cuh>

#include "uninitialized_copy.cuh"

/**
 * Reduces a thread-local array from right to left (suffix reduction).
 */
template<int LENGTH, typename T, typename ReductionOp>
__device__ __forceinline__
T ThreadReverseReduce(const T (&input)[LENGTH], ReductionOp op) {
    static_assert(LENGTH > 0);

    T aggregate = input[LENGTH - 1];

    #pragma  unroll
    for (int i = LENGTH - 2; i >= 0; i--) {
        aggregate = op(aggregate, input[i]);
    }

    return aggregate;
}


/**
 * Reverse inclusive scan over a thread-local array with a postfix seed.
 */
template<int LENGTH, typename T, typename ScanOp>
__device__ __forceinline__
T ThreadReverseScanInclusive(const T (&input)[LENGTH], T (&output)[LENGTH], ScanOp op, const T postfix) {
    static_assert(LENGTH > 0);

    T aggregate = postfix;

    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; i--) {
        aggregate = op(aggregate, input[i]);
        output[i] = aggregate;
    }

    return aggregate;
}

/**
 * Reverse exclusive scan over a thread-local array with a postfix seed.
 */
template<int LENGTH, typename T, typename ScanOp>
__device__ __forceinline__
T ThreadReverseScanExclusive(const T (&input)[LENGTH], T (&output)[LENGTH], ScanOp op, const T postfix) {
    static_assert(LENGTH > 0);

    T aggregate = postfix;

    #pragma unroll
    for (int i = LENGTH - 1; i >= 0; i--) {
        output[i] = aggregate;
        aggregate = op(aggregate, input[i]);
    }

    return aggregate;
}


/**
 * Warp-level reverse scan (suffix scan) over LOGICAL_WARP_THREADS lanes.
 */
template<typename T, int LOGICAL_WARP_THREADS>
struct WarpReverseScan {
    static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == CUB_WARP_THREADS(0));
    static constexpr int STEPS = cub::Log2<LOGICAL_WARP_THREADS>::VALUE;
    static_assert(LOGICAL_WARP_THREADS == 1 << STEPS);

    unsigned int lane_id;
    unsigned int warp_id;
    unsigned int member_mask;

    explicit __device__ __forceinline__
    WarpReverseScan()
        : lane_id(threadIdx.x & 0x1f)
        , warp_id(IS_ARCH_WARP ? 0 : (lane_id / LOGICAL_WARP_THREADS))
        , member_mask(cub::WarpMask<LOGICAL_WARP_THREADS>(warp_id))
    {
        if constexpr (!IS_ARCH_WARP) {
            lane_id = lane_id % LOGICAL_WARP_THREADS;
        }
    }

    /**
     * Broadcast a value from src_lane to all lanes in the warp.
     */
    __device__ __forceinline__
    T Broadcast(T input, int src_lane) {
        return cub::ShuffleIndex<LOGICAL_WARP_THREADS>(input, src_lane, member_mask);
    }

    /**
     * Warp-wide reverse inclusive scan (suffix propagation via shuffle-down).
     */
    template<typename ScanOp>
    __device__ __forceinline__
    void InclusiveReverseScan(T input, T& aggregate, ScanOp op) {
        aggregate = input;

        // Parallel suffix propagation using shuffle-down steps
        #pragma unroll
        for (int step = 0; step < STEPS; step++) {
            int offset = 1 << step;
            T peer = cub::ShuffleDown<LOGICAL_WARP_THREADS>(
                aggregate, offset, LOGICAL_WARP_THREADS - 1, member_mask
            );
            aggregate = static_cast<int>(lane_id) >= LOGICAL_WARP_THREADS - offset
                ? aggregate
                : op(peer, aggregate);
        }
    }

    /**
     * Warp-wide reverse exclusive scan; also outputs total warp reduction.
     */
    template<typename ScanOp>
    __device__ __forceinline__
    void ExclusiveReverseScan(T input, T& aggregate, ScanOp op, T& warp_aggregate) {
        T inclusive_aggregate;
        InclusiveReverseScan(input, inclusive_aggregate, op);

        // Extract total warp sum and shift lanes to convert inclusive to exclusive
        warp_aggregate = cub::ShuffleIndex<LOGICAL_WARP_THREADS>(inclusive_aggregate, 0, member_mask);

        aggregate = cub::ShuffleDown<LOGICAL_WARP_THREADS>(inclusive_aggregate, 1, LOGICAL_WARP_THREADS - 1, member_mask);
    }

    /**
     * Compute both reverse inclusive and exclusive scans for the warp.
     */
    template<typename ScanOp>
    __device__ __forceinline__
    void ReverseScan(T input, T& inclusive_aggregate, T& exclusive_aggregate, ScanOp op) {
        InclusiveReverseScan(input, inclusive_aggregate, op);
        exclusive_aggregate = cub::ShuffleDown<LOGICAL_WARP_THREADS>(inclusive_aggregate, 1, LOGICAL_WARP_THREADS - 1, member_mask);
    }
};

/**
 * Block-wide reverse scan using raking reduction in shared memory.
 */
template<typename T, int BLOCK_DIM_X, bool MEMOIZE=false>
struct BlockReverseScan {
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X;
    using BlockRakingLayout = cub::BlockRakingLayout<T, BLOCK_THREADS>;
    static_assert(BlockRakingLayout::UNGUARDED);
    static constexpr int RAKING_THREADS = BlockRakingLayout::RAKING_THREADS;
    static constexpr int SEGMENT_LENGTH = BlockRakingLayout::SEGMENT_LENGTH;
    static constexpr bool WARP_SYNCHRONOUS = (int(BLOCK_THREADS) == int(RAKING_THREADS));

    using WarpReverseScan = WarpReverseScan<T, RAKING_THREADS>;

    struct _TempStorage {
        typename BlockRakingLayout::TempStorage raking_grid;
    };
    
    struct TempStorage : cub::Uninitialized<_TempStorage> {};

    _TempStorage& temp_storage;
    unsigned int linear_tid;
    T cached_segment[SEGMENT_LENGTH];

    /**
     * Reduce each shared-memory segment to a single aggregate.
     */
    template<typename ScanOp>
    __device__ __forceinline__
    T UpSweep(ScanOp op) {
        T* smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);

        #pragma unroll
        for (int i = 0;i < SEGMENT_LENGTH; i++) {
            cached_segment[i] = smem_raking_ptr[i];
        }
        
        T segment_aggregate = cached_segment[SEGMENT_LENGTH - 1];

        #pragma unroll
        for (int i = SEGMENT_LENGTH - 2; i >= 0; i--) {
            segment_aggregate = op(segment_aggregate, cached_segment[i]);
        }

        return segment_aggregate;
    }

    /**
     * Apply segment postfix and write results back to shared memory.
     */
    template<typename ScanOp>
    __device__ __forceinline__
    void ExclusiveDownSweep(ScanOp op, T postfix) {
        T* smem_raking_ptr = BlockRakingLayout::RakingPtr(temp_storage.raking_grid, linear_tid);
        if (!MEMOIZE) {
            #pragma unroll
            for (int i = 0; i < SEGMENT_LENGTH; i++) {
                cached_segment[i] = smem_raking_ptr[i];
            }
        }
        ThreadReverseScanExclusive(cached_segment, cached_segment, op, postfix);
        #pragma unroll
        for (int i = 0; i < SEGMENT_LENGTH; i++) {
            smem_raking_ptr[i] = cached_segment[i];
        }
    }

    __device__ __forceinline__
    BlockReverseScan(TempStorage& temp_storage)
        : temp_storage(temp_storage.Alias())
        , linear_tid(cub::RowMajorTid(BLOCK_DIM_X, 1, 1))
    { }

    /**
     * Entry point for block-wide reverse exclusive scan.
     */
    template<typename ScanOp, typename BlockCarryInOp>
    __device__ __forceinline__
    void ExclusiveReverseScan(T input, T& exclusive_aggregate, ScanOp scan_op, BlockCarryInOp& block_carry_in_op) {
        if (WARP_SYNCHRONOUS) {
            // Small blocks process scan within a single warp

            T block_aggregate;
            WarpReverseScan warp_scan;
            warp_scan.ExclusiveReverseScan(input, exclusive_aggregate, scan_op, block_aggregate);

            // Acquire global postfix and broadcast to all threads
            T block_postfix = block_carry_in_op(block_aggregate);
            block_postfix = warp_scan.Broadcast(block_postfix, 0);

            // Merge block-level postfix into local results
            exclusive_aggregate = linear_tid == BLOCK_TRHEADS - 1
                ? block_postfix
                : scan_op(block_postfix, exclusive_aggregate);
        }
        else {
            // Multi-warp blocks use shared memory raking

            T* placement_ptr = BlockRakingLayout::PlacementPtr(temp_storage.raking_grid, linear_tid);
            detail::uninitialized_copy(placement_ptr, input);
            __syncthreads();
            if (linear_tid < RAKING_THREADS) {
                // Raking threads compute the cross-segment scan

                WarpReverseScan warp_scan;
                T segment_aggregate = UpSweep(scan_op);
                T segment_exclusive_aggregate, block_aggregate;
                warp_scan.ExclusiveReverseScan(segment_aggregate, segment_exclusive_aggregate, scan_op, block_aggregate);
                
                // Integrate external block carry-in
                T block_postfix = block_carry_in_op(block_aggregate);
                block_postfix = warp_scan.Broadcast(block_postfix, 0);

                // Calculate and distribute the specific postfix for each thread's segment
                T segment_postfix = linear_tid == RAKING_THREADS - 1
                    ? block_postfix
                    : scan_op(block_postfix, segment_exclusive_aggregate);

                ExclusiveDownSweep(scan_op, segment_postfix);
            }
            __syncthreads();
            exclusive_aggregate = *placement_ptr;
        }
    }

    /**
     * Block-wide reverse inclusive scan (multiple items per thread).
     */
    template<int ITEMS_PER_THREAD, typename ScanOp, typename BlockCarryInOp>
    __device__ __forceinline__
    void InclusiveReverseScan(T (&input)[ITEMS_PER_THREAD], T (&output)[ITEMS_PER_THREAD], ScanOp scan_op, BlockCarryInOp& block_carry_in_op) {
        // Local reduction -> Block-wide exclusive scan -> Local inclusive scan
        T postfix = ThreadReverseReduce(input, scan_op);
        ExclusiveReverseScan(postfix, postfix, scan_op, block_carry_in_op);
        ThreadReverseScanInclusive(input, output, scan_op, postfix);
    }
};