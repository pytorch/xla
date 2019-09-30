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

at::ScalarType GetScalarType(at::Scalar scalar) {
  if (scalar.isFloatingPoint()) {
    return at::kDouble;
  } else if (scalar.isBoolean()) {
    return at::kBool;
  } else if (scalar.isComplex()) {
    return at::kComplexDouble;
  } else {
    XLA_CHECK(scalar.isIntegral(/*includeBool=*/false));
    return at::kLong;
  }
}
}  // namespace torch_xla
