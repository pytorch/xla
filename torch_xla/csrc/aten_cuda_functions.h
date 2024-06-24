#ifndef XLA_TORCH_XLA_CSRC_ATEN_CUDA_FUNCTIONS_H_
#define XLA_TORCH_XLA_CSRC_ATEN_CUDA_FUNCTIONS_H_

#include <cstdint>

// Forward declaration of PyTorch CUDA functions.
// Source: c10/cuda/CUDAFunctions.h
//
// These are needed in order to synchronize the CUDA device after running
// the operation in PyTorch eager mode.
//
// It would be better to include the actual header. However, if we build
// PyTorch/XLA in an environment where PyTorch wasn't compiled with CUDA
// (i.e. our CI), the build would fail.

namespace c10 {

// Type alias used inside PyTorch.
using DeviceIndex = int8_t;

namespace cuda {

c10::DeviceIndex current_device() noexcept;
void set_device(c10::DeviceIndex);
void device_synchronize();

}  // namespace cuda
}  // namespace c10

#endif  // XLA_TORCH_XLA_CSRC_ATEN_CUDA_FUNCTIONS_H_
