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