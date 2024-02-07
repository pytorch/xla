#include "torch_xla/csrc/ops/embedding_bag.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {
xla::Shape NodeOutputShape(const torch::lazy::Value& weight,
                           const torch::lazy::Value& indices,
                           const torch::lazy::Value& offsets) {
  xla::Shape weight_shape = GetXlaShape(weight);
  xla::Shape indices_shape = GetXlaShape(indices);
  xla::Shape offsets_shape = GetXlaShape(offsets);
  std::vector<int64_t> dimensions;
  if (indices_shape.rank() == 2) {
    dimensions.push_back(indices_shape.dimensions(0));
  } else {
    dimensions.push_back(offsets_shape.dimensions(0));
  }
  dimensions.push_back(weight_shape.dimensions(1));
  return weight_shape;
  // return xla::ShapeUtil::MakeShape(weight_shape.element_type(), dimensions);
}

}  // namespace

std::string EmbeddingBag::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  return ss.str();
}

EmbeddingBag::EmbeddingBag(const torch::lazy::Value& weight,
                           const torch::lazy::Value& indices,
                           const torch::lazy::Value& offsets,
                           bool scale_grad_by_freq, int64_t mode, bool sparse,
                           const c10::optional<at::Tensor>& per_sample_weights,
                           bool include_last_offset, int64_t padding_idx)
    : XlaNode(
          torch::lazy::OpKind(at::aten::embedding_bag),
          {weight, indices, offsets, per_sample_weights},
          [&]() { return NodeOutputShape(weight, indices, offsets); },
          /*num_outputs=*/1,
          torch::lazy::MHash(scale_grad_by_freq, mode, sparse,
                             include_last_offset, padding_idx)),
      scale_grad_by_freq_(scale_grad_by_freq),
      mode_(mode),
      sparse_(sparse),
      include_last_offset_(include_last_offset),
      padding_idx_(padding_idx) {}

torch::lazy::NodePtr EmbeddingBag::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<EmbeddingBag>(operands.at(0), operands.at(0),
                                             operands.at(0), false, 1, false,
                                             c10::nullopt, false, 0);
}

XlaOpVector EmbeddingBag::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output1 = xla::Log(input);
  xla::XlaOp output2 = xla::Log(input);
  xla::XlaOp output3 = xla::Log(input);
  xla::XlaOp output4 = xla::Log(input);
  return ReturnOps({output1, output2, output3, output4}, loctx);
}

}  // namespace torch_xla