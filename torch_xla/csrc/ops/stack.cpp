#include "torch_xla/csrc/ops/stack.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(c10::ArrayRef<torch::lazy::Value> values,
                           int64_t dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildStack(operands, dim);
  };
  std::vector<xla::Shape> shapes;
  shapes.reserve(values.size());
  for (auto& value : values) {
    shapes.push_back(GetXlaShape(value));
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

Stack::Stack(c10::ArrayRef<torch::lazy::Value> values, int64_t dim)
    : XlaNode(
          torch::lazy::OpKind(at::aten::stack), values,
          [&]() { return NodeOutputShape(values, dim); },
          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr Stack::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Stack>(operands, dim_);
}

XlaOpVector Stack::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  return ReturnOp(BuildStack(inputs, dim_), loctx);
}

std::string Stack::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
