#pragma once

#include <cstddef>
#include <initializer_list>

constexpr size_t max_of(std::initializer_list<size_t> list) {
    auto it = list.begin();
    size_t max_val = *it++;
    for (; it != list.end(); it++) {
        if (*it > max_val) {
            max_val = *it;
        }
    }
    return max_val;
}

template<typename Traits>
inline __device__
void load(
    float* src, 
    float (&dst)[Traits::kNumElements], 
    typename Traits::ScalarBlockLoad::TempStorage& smem,
    int num_valid_elements
) {
    if constexpr (Traits::kSeqDivisible) {
        auto& smem_vec = reinterpret_cast<typename Traits::VectorBlockLoad::TempStorage&>(smem);
        typename Traits::VectorBlockLoad(smem_vec).Load(
            reinterpret_cast<float4*>(src),
            reinterpret_cast<float4(&)[Traits::kNumVectors]>(dst)
        );
    }
    else {
        typename Traits::ScalarBlockLoad(smem).Load(src, dst, num_valid_elements, 0.f);
    }
}

template<typename Traits>
inline __device__
void store(
    const float (&src)[Traits::kNumElements],
    float* dst,
    typename Traits::ScalarBlockStore::TempStorage& smem,
    int num_valid_elements
) {
    float vals[Traits::kNumElements];

    #pragma unroll
    for (int i = 0; i < Traits::kNumElements; i++) {
        vals[i] = src[i];
    }
    
    if constexpr (Traits::kSeqDivisible) {
        auto& smem_vec = reinterpret_cast<typename Traits::VectorBlockStore::TempStorage&>(smem);
        typename Traits::VectorBlockStore(smem_vec).Store(
            reinterpret_cast<float4*>(dst),
            reinterpret_cast<float4(&)[Traits::kNumVectors]>(vals)
        );
    }
    else {
        typename Traits::ScalarBlockStore(smem).Store(dst, vals, num_valid_elements);
    }
}

struct ScanOp {
    __device__ __forceinline__
    float2 operator()(const float2& ab0, const float2& ab1) const {
        float a0 = ab0.x;
        float b0 = ab0.y;
        float a1 = ab1.x;
        float b1 = ab1.y;
        return make_float2(a1 * a0, a1 * b0 + b1);
    }
};

struct StateCarryCallbackOp {
    float2 carry;

    explicit __device__
    StateCarryCallbackOp(float2 init): carry(init) { }

    __device__
    float2 operator()(float2 aggregate) {
        float2 old = carry;
        carry = ScanOp()(carry, aggregate);
        return old;
    }
};