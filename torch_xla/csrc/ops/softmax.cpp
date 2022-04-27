#include "torch_xla/csrc/ops/softmax.h"

#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/softmax_builder.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerSoftmax(xla::XlaOp input, int64_t dim,
                        const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildSoftmax(input, dim);
  return CastToScalarType(result, dtype);
}

xla::Shape NodeOutputShape(const XlaValue& input,
                           const c10::optional<at::ScalarType>& dtype) {
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        input.xla_shape(), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.xla_shape();
}

}  // namespace

Softmax::Softmax(const XlaValue& input, int64_t dim,
                 c10::optional<at::ScalarType> dtype)
    : XlaNode(torch::lazy::OpKind(at::aten::softmax), {input},
              [&]() { return NodeOutputShape(input, dtype); },
              /*num_outputs=*/1,
              torch::lazy::MHash(dim, torch::lazy::OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {}

torch::lazy::NodePtr Softmax::Clone(OpList operands) const {
  return ir::MakeNode<Softmax>(operands.at(0), dim_, dtype_);
}

XlaOpVector Softmax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerSoftmax(input, dim_, dtype_), loctx);
}

std::string Softmax::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_
     << ", dtype=" << torch::lazy::OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
