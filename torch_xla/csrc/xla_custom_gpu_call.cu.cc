#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream> 

__global__ custom_gpu_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

// CPU function responsible for launching a CUDA kernel.
void do_custom_gpu_call(CUstream stream, void** buffers,
                    const char* opaque, size_t opaque_len) {
  const float* in0 = reinterpret_cast<const float*>(buffers[0]);
  const float* in1 = reinterpret_cast<const float*>(buffers[1]);
  float* out = reinterpret_cast<float*>(buffers[2]);

  const int64_t block_dim = 64;
  const int64_t grid_dim = 2048 / block_dim;
  std::cout << "Calling GPU kernel" << std::endl;
  custom_gpu_kernel<<<grid_dim, block_dim, 0, stream>>>(in0, in1, out);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_gpu_call, "CUDA");


