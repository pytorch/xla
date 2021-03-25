#pragma once

namespace tensorflow {

enum TensorFormat {
  FORMAT_NHWC = 0,
  FORMAT_NCHW = 1,
  FORMAT_NCHW_VECT_C = 2,
  FORMAT_NHWC_VECT_W = 3,
  FORMAT_HWNC = 4,
  FORMAT_HWCN = 5,
};

}  // namespace tensorflow
