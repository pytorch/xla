#include "torch_xla/csrc/ops/unsqueeze.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, int dim) {
  const xla::Shape& shape = input.shape();
  auto dimensions = BuildUnsqueezeDimensions(shape.dimensions(), dim);
  return xla::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

}  // namespace

Unsqueeze::Unsqueeze(const Value& input, int dim)
    : Node(
          ir::OpKind(at::aten::unsqueeze), {input},
          [&]() { return NodeOutputShape(input, dim); },
          /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Unsqueeze::Clone(OpList operands) const {
  return MakeNode<Unsqueeze>(operands.at(0), dim_);
}

XlaOpVector Unsqueeze::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildUnsqueeze(input, dim_);
  return ReturnOp(output, loctx);
}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
