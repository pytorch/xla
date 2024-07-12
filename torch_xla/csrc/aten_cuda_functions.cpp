#include "torch_xla/csrc/aten_cuda_functions.h"

#include <pybind11/pybind11.h>

#include <cassert>
#include <stdexcept>

static void fail(const char* name) {
  throw std::runtime_error("PyTorch was compiled without CUDA support.");
}

namespace c10::cuda {

c10::DeviceIndex current_device() { fail("c10::cuda::current_device()"); }

void set_device(c10::DeviceIndex) { fail("c10::cuda::set_device()"); }

void device_synchronize() { fail("c10::cuda::device_synchronize()"); }

}  // namespace c10::cuda

PYBIND11_MODULE(_XLAC_cuda_functions, m) {}
