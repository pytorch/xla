#include "torch_xla/csrc/xla_custom_call.h"

#include <iostream> 


// TODO: number of inputs and shapes of inputs and outputs are
// hardcoded. Need to be sent in the args. 
// TODO: template the datatype?
void argmin_custom(void* out, const void** in) {
  int* out_buf = reinterpret_cast<int*>(out);
  const int* in0 = reinterpret_cast<const int*>(in[0]);
  int min_idx = 0;
  std::cout << "Custom call entered - Assumes only 4 contiguous elements" << std::endl;
  for (int i = 0; i < 4; ++i) {
    std::cout << "in[" << i << "]=" << in0[i] << std::endl;
    if (in0[i] < in0[min_idx]) {
      min_idx = i;
    }
  }
  out_buf[0] = min_idx;
}


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


