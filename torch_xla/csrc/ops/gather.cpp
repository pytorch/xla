#include "torch_xla/csrc/ops/gather.h"

#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& index, int64_t dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::TorchGather(operands[0], operands[1], dim,
                            IsSparseGather(operands[0], operands[1], dim));
  };

  return InferOutputShape({GetXlaShape(input), GetXlaShape(index)},
                          lower_for_shape_fn);
}

}  // namespace

Gather::Gather(const torch::lazy::Value& input, int64_t dim,
               const torch::lazy::Value& index)
    : XlaNode(
          torch::lazy::OpKind(at::aten::gather), {input, index},
          [&]() { return NodeOutputShape(input, index, dim); },
          /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr Gather::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Gather>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector Gather::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      xla::TorchGather(input, index, dim_, IsSparseGather(input, index, dim_)),
      loctx);
}

std::string Gather::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
