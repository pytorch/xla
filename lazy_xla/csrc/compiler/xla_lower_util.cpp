#include "lazy_xla/csrc/compiler/xla_lower_util.h"

#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"

namespace torch_lazy_tensors {
namespace compiler {

XlaOpVector BuildAmpForeachNonFiniteCheckAndUnscale(
    const XlaOpVector& inputs, const xla::XlaOp& found_inf_float,
    const xla::XlaOp& inv_scale) {
  const xla::PrimitiveType origin_type =
      XlaHelpers::ShapeOfXlaOp(found_inf_float).element_type();
  xla::XlaOp one = xla::One(inputs[0].builder(), xla::PrimitiveType::PRED);
  xla::XlaOp found_inf =
      xla::ConvertElementType(found_inf_float, xla::PrimitiveType::PRED);
  for (size_t i = 0; i < inputs.size(); ++i) {
    xla::XlaOp all_finite =
        xla::ReduceAll(xla::IsFinite(inputs[i]), one,
                       xla::CreateScalarAndComputation(xla::PrimitiveType::PRED,
                                                       inputs[i].builder()));
    found_inf = xla::Or(found_inf, xla::Not(all_finite));
  }
  XlaOpVector results;
  for (size_t i = 0; i < inputs.size(); ++i) {
    results.push_back(inputs[i] * inv_scale);
  }
  results.push_back(xla::ConvertElementType(found_inf, origin_type));
  return results;
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
