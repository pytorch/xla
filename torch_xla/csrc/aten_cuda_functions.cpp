#include "torch_xla/csrc/aten_cuda_functions.h"

#include <pybind11/pybind11.h>

#include <cassert>
#include <stdexcept>

// Context
// =======
// aten_fallback.cpp (compiled into _XLAC.so library) uses these functions
// for providing OpenXLA fallback on CUDA. Therefore, they must be defined at
// some point, somewhere.
//
// Problem
// =======
// These functions' definition should be provided by PyTorch. However, whenever
// it's not compiled with CUDA support, it doesn't.
//
// Solution
// ========
// Load these backup definitions, whenever we detect that the PyTorch being
// used doesn't support CUDA.
//
// More Context
// ============
// Our CI currently compiles PyTorch/XLA only once. However, the CI uses it
// with two versions of PyTorch: compiled with and without CUDA support.
// Thus, we must find a way of, at runtime, conditionally load these functions'
// definition, given the current used PyTorch.
//
// This file is compiled into a additional Python library that is conditionally
// loaded given the result of torch.cuda.is_available().

static void fail(const char* name) {
  throw std::runtime_error("PyTorch was compiled without CUDA support.");
}

namespace c10::cuda {

// Returning 0 in this function forces OpenXLA fallback execution on CPU.
DeviceIndex device_count() noexcept { return 0; }

c10::DeviceIndex current_device() { fail("c10::cuda::current_device()"); }

void set_device(c10::DeviceIndex) { fail("c10::cuda::set_device()"); }

void device_synchronize() { fail("c10::cuda::device_synchronize()"); }

}  // namespace c10::cuda

PYBIND11_MODULE(_XLAC_cuda_functions, m) {}
