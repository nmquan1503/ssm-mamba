#include <torch/extension.h>

#include "selective_scan.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selective_scan_forward, "Selective scan forward");
    m.def("backward", &selective_scan_backward, "Selective scan backward");
}