#include "torch_xla/csrc/ops/amin.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           std::vector<int64_t>& dimensions, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMinInDims(operands[0], dimensions, keepdim);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Amin::Amin(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
           bool keepdim)
    : XlaNode(torch::lazy::OpKind(at::aten::amin), {input},
              [&]() { return NodeOutputShape(input, dimensions, keepdim); },
              /*num_outputs=*/1, torch::lazy::MHash(dimensions, keepdim)),
      dimensions_(std::move(dimensions)),
      keepdim_(keepdim) {}

torch::lazy::NodePtr Amin::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Amin>(operands.at(0), dimensions_, keepdim_);
}

XlaOpVector Amin::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildMinInDims(input, dimensions_, keepdim_), loctx);
}

std::string Amin::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", dimensions=" << absl::StrJoin(dimensions_, ", ")
     << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace torch_xla
