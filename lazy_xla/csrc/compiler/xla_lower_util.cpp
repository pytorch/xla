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

XlaOpVector BuildAmpUpdateScale(const xla::XlaOp& growth_tracker,
                                const xla::XlaOp& current_scale,
                                const xla::XlaOp& found_inf_float,
                                double scale_growth_factor,
                                double scale_backoff_factor,
                                int scale_growth_interval) {
  xla::XlaOp one = xla::One(growth_tracker.builder(), xla::PrimitiveType::S32);
  xla::XlaOp one_float =
      xla::One(growth_tracker.builder(), xla::PrimitiveType::F32);
  xla::XlaOp found_inf =
      xla::ConvertElementType(found_inf_float, xla::PrimitiveType::PRED);
  const auto& growth_factor = XlaHelpers::ScalarValue<float>(
      scale_growth_factor,
      XlaHelpers::ShapeOfXlaOp(current_scale).element_type(),
      growth_tracker.builder());
  const auto& backoff_factor = XlaHelpers::ScalarValue<float>(
      scale_backoff_factor,
      XlaHelpers::ShapeOfXlaOp(current_scale).element_type(),
      growth_tracker.builder());
  const auto& growth_interval = XlaHelpers::ScalarValue<int>(
      scale_growth_interval,
      XlaHelpers::ShapeOfXlaOp(growth_tracker).element_type(),
      growth_tracker.builder());

  xla::XlaOp all_finite = xla::Not(found_inf);
  xla::XlaOp not_achieve_interval = xla::ConvertElementType(
      growth_interval - one - growth_tracker, xla::PrimitiveType::PRED);
  xla::XlaOp new_growth_tracker =
      (growth_tracker + one) *
      ConvertElementType(xla::And(all_finite, not_achieve_interval),
                         xla::PrimitiveType::S32);
  xla::XlaOp growth_factor_or_one = xla::Max(
      growth_factor * xla::ConvertElementType(
                          xla::And(all_finite, xla::Not(not_achieve_interval)),
                          xla::PrimitiveType::F32),
      one_float);
  xla::XlaOp backoff_factor_or_one =
      backoff_factor *
          xla::ConvertElementType(found_inf, xla::PrimitiveType::F32) +
      xla::ConvertElementType(all_finite, xla::PrimitiveType::F32);
  xla::XlaOp new_scale =
      current_scale * growth_factor_or_one * backoff_factor_or_one;
  return {new_growth_tracker, new_scale};
}

}  // namespace compiler
}  // namespace torch_lazy_tensors
