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


