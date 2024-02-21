#include "torch_xla/csrc/ops/cat.h"

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(c10::ArrayRef<torch::lazy::Value> values,
                           int64_t dim, at::ScalarType dtype) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildCat(operands, dim, dtype);
  };
  std::vector<xla::Shape> shapes;
  shapes.reserve(values.size());
  for (auto& value : values) {
    shapes.push_back(GetXlaShape(value));
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

Cat::Cat(c10::ArrayRef<torch::lazy::Value> values, int64_t dim,
         at::ScalarType dtype)
    : XlaNode(
          torch::lazy::OpKind(at::aten::cat), values,
          [&]() { return NodeOutputShape(values, dim, dtype); },
          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim),
      dtype_(dtype) {}

torch::lazy::NodePtr Cat::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Cat>(operands, dim_, dtype_);
}

XlaOpVector Cat::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (auto& operand : operands()) {
    inputs.push_back(loctx->GetOutputOp(operand));
  }
  return ReturnOp(BuildCat(inputs, dim_, dtype_), loctx);
}

std::string Cat::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_ << ", dtype=" << dtype_;
  return ss.str();
}

}  // namespace torch_xla
