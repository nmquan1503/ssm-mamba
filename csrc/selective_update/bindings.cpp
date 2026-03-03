#include <torch/extension.h>

#include "selective_update.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply", &selective_update, "Selective update");
}