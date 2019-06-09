#include "torch_xla/csrc/ops/cast.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, at::ScalarType dtype) {
  xla::Shape shape = input.shape();
  shape.set_element_type(MakeXlaPrimitiveType(dtype, /*device=*/nullptr));
  return shape;
}

}  // namespace

Cast::Cast(const Value& input, at::ScalarType dtype)
    : Node(xla_cast, {input}, NodeOutputShape(input, dtype),
           /*num_outputs=*/1, xla::util::MHash(static_cast<int>(dtype))),
      dtype_(dtype) {}

NodePtr Cast::Clone(OpList operands) const {
  return MakeNode<Cast>(operands.at(0), dtype_);
}

XlaOpVector Cast::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      ConvertTo(input, operand(0).shape().element_type(),
                MakeXlaPrimitiveType(dtype_, /*device=*/nullptr),
                /*device=*/nullptr);
  return ReturnOp(output, loctx);
}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dtype=" << dtype_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
