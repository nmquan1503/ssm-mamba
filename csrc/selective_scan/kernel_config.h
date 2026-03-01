#pragma once

namespace kernel_config {
    constexpr int num_threads = 32;
    constexpr int num_elements = 4;
    constexpr int chunk_size = num_threads * num_elements;

    inline int get_num_chunks(const int seq_len) {
        return (seq_len - 1) / chunk_size + 1;
    }
}