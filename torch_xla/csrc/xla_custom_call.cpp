#include "torch_xla/csrc/xla_custom_call.h"

#include <iostream> 


// TODO: number of inputs and shapes of inputs and outputs are
// hardcoded. Need to be sent in the args. 
void argmin_custom(void* out, const void** in) {
  int* out_buf = reinterpret_cast<int*>(out);
  const float* in0 = reinterpret_cast<const float*>(in[0]);
  int min_idx = 0;
  std::cout << "Custom call entered" << std::endl;
  for (int i = 0; i < 4; ++i) {
    std::cout << "in[" << i << "]=" << in0[i] << std::endl;
    if (in0[i] < in0[min_idx]) {
      min_idx = i;
    }
  }
  std::cout << "Computation is " << min_idx << std::endl;
  out_buf[0] = min_idx;
}


