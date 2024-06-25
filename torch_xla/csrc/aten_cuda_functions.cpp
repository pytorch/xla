#include "torch_xla/csrc/aten_cuda_functions.h"

#include <c10/util/Exception.h>
#include <pybind11/pybind11.h>

static void fail(const char* name) {
  TORCH_CHECK(false, "Could not call the CUDA function: ", name,
              ". PyTorch was compiled without CUDA support.");
}

namespace c10::cuda {

c10::DeviceIndex current_device() noexcept { return -1; }

void set_device(c10::DeviceIndex) { fail("c10::cuda::set_device()"); }

void device_synchronize() { fail("c10::cuda::device_synchronize()"); }

}  // namespace c10::cuda

PYBIND11_MODULE(_XLAC_cuda_functions, m) {}
