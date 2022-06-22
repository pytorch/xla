#include <torch_xla/csrc/generated/LazyIr.h>

#include "tensorflow/compiler/xla/client/lib/logdet.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/matrix.h"

namespace torch_xla {

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

torch_xla::XlaOpVector Erf::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Erf(xla_input), loctx);
}

torch_xla::XlaOpVector Erfc::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Erfc(xla_input), loctx);
}

torch_xla::XlaOpVector Erfinv::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::ErfInv(xla_input), loctx);
}

torch_xla::XlaOpVector Exp::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Exp(xla_input), loctx);
}

torch_xla::XlaOpVector Floor::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Floor(xla_input), loctx);
}

torch_xla::XlaOpVector Inverse::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildInverse(xla_input), loctx);
}

torch_xla::XlaOpVector Logdet::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::LogDet(xla_input), loctx);
}

torch_xla::XlaOpVector Maximum::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  auto promoted = XlaHelpers::Promote(xla_input, xla_other);
  return ReturnOp(xla::Max(promoted.first, promoted.second), loctx);
}

torch_xla::XlaOpVector Minimum::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  xla::XlaOp xla_other = loctx->GetOutputOp(operand(1));
  auto promoted = XlaHelpers::Promote(xla_input, xla_other);
  return ReturnOp(xla::Min(promoted.first, promoted.second), loctx);
}

torch_xla::XlaOpVector Reciprocal::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildReciprocal(xla_input), loctx);
}

torch_xla::XlaOpVector Sgn::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildSgn(xla_input), loctx);
}

torch_xla::XlaOpVector Sign::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildSign(xla_input), loctx);
}

torch_xla::XlaOpVector Sin::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Sin(xla_input), loctx);
}

torch_xla::XlaOpVector Sinh::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Sinh(xla_input), loctx);
}

/* Blocked on https://github.com/pytorch/xla/issues/3596 */
// torch_xla::XlaOpVector Slogdet::Lower(LoweringContext* loctx) const {
//   xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
//   xla::SignAndLogDet result = xla::SLogDet(xla_input);
//   return ReturnOps({result.sign, result.logdet}, loctx);
// }

torch_xla::XlaOpVector Tan::Lower(LoweringContext* loctx) const {
  xla::XlaOp xla_input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::Tan(xla_input), loctx);
}

}  // namespace torch_xla
