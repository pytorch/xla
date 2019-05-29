#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

at::Tensor CopyTensor(const at::Tensor& ref) {
  return ref.to(ref.options(), /*non_blocking=*/false, /*copy=*/true);
}

// Same as above, with an additional cast.
at::Tensor CopyTensor(const at::Tensor& ref, at::ScalarType dest_type) {
  return ref.to(ref.options().dtype(dest_type), /*non_blocking=*/false,
                /*copy=*/true);
}

at::Tensor ToTensor(const at::Tensor& tensor) {
  return tensor.is_variable()
             ? torch::autograd::as_variable_ref(tensor).tensor_data()
             : tensor;
}

}  // namespace torch_xla
