#include "torch_xla/csrc/xla_custom_call.h"
#include <iostream> 


__global__ custom_gpu_kernel(const float* in0, const float* in1, float* out) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_gpu_call(CUstream stream, void** buffers,
                    const char* opaque, size_t opaque_len) {
  const float* in0 = reinterpret_cast<const float*>(buffers[0]);
  const float* in1 = reinterpret_cast<const float*>(buffers[1]);
  float* out = reinterpret_cast<float*>(buffers[2]);

  const int64_t block_dim = 64;
  const int64_t grid_dim = 2048 / block_dim;
  custom_gpu_kernel<<<grid_dim, block_dim,
                       /*dynamic_shared_mem_bytes=*/0, stream>>>(in0, in1, out);
}


// "@local_config_cuda//cuda:cuda_headers", # cuda
// "@local_config_cuda//cuda:cudart", # cuda runtime