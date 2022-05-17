#include <torch_xla/csrc/generated/LazyIr.h>

#include "tensorflow/compiler/xla/client/lib/math.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

// If the XlaOp is not a floating point, cast it to float_type.
xla::XlaOp GetFloatingOp(const xla::XlaOp& input,
                         xla::PrimitiveType float_type) {
  if (xla::primitive_util::IsIntegralType(XlaHelpers::TypeOfXlaOp(input))) {
    const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
    xla::PrimitiveType raw_from = input_shape.element_type();
    xla::PrimitiveType raw_to = float_type;
    xla::XlaOp input = ConvertToRaw(input, raw_from, raw_from, raw_to, raw_to,
                                    /*device=*/nullptr);
  }
  return input;
}

}  // namespace

torch_xla::XlaOpVector Abs::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAbs(xla_input), loctx);
}

torch_xla::XlaOpVector Acos::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Acos(xla_input), loctx);
}

torch_xla::XlaOpVector Acosh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Acosh(xla_input), loctx);
}

torch_xla::XlaOpVector Asin::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Asin(xla_input), loctx);
}

torch_xla::XlaOpVector Asinh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Asinh(xla_input), loctx);
}

torch_xla::XlaOpVector Atan::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Atan(xla_input), loctx);
}

torch_xla::XlaOpVector Atanh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Atanh(xla_input), loctx);
}

torch_xla::XlaOpVector Cos::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Cos(xla_input), loctx);
}

torch_xla::XlaOpVector Cosh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Cosh(xla_input), loctx);
}
torch_xla::XlaOpVector Log::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Log(xla_input), loctx);
}

torch_xla::XlaOpVector Log2::Lower(LoweringContext* loctx) const {
  double base = 2.0;
  xla::XlaOp xla_input =
      GetFloatingOp(loctx->GetOutputOp(operand(0)), xla::PrimitiveType::F32);
  return ReturnOp(BuildLogBase(xla_input, base), loctx);
}

torch_xla::XlaOpVector Log10::Lower(LoweringContext* loctx) const {
  double base = 10.0;
  xla::XlaOp xla_input =
      GetFloatingOp(loctx->GetOutputOp(operand(0)), xla::PrimitiveType::F32);
  return ReturnOp(BuildLogBase(xla_input, base), loctx);
}

torch_xla::XlaOpVector Maximum::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  auto promoted = XlaHelpers::Promote(xla_input, xla_other);
  return ReturnOp(xla::Max(promoted.first, promoted.second), loctx);
}

torch_xla::XlaOpVector Sgn::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildSgn(xla_input), loctx);
}

torch_xla::XlaOpVector Sign::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildSign(xla_input), loctx);
}

}  // namespace torch_xla
