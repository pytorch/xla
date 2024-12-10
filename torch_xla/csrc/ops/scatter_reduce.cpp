#include "torch_xla/csrc/ops/scatter_reduce.h"

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

ScatterReduce::ScatterReduce(const torch::lazy::Value& input,
                             const torch::lazy::Value& index,
                             const torch::lazy::Value& src,
                             std::string_view reduce, bool include_self,
                             int64_t dim)
    : XlaNode(torch::lazy::OpKind(at::aten::scatter_reduce),
              {input, index, src}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(dim, reduce, include_self)),
      reduce_(reduce),
      include_self_(include_self),
      dim_(dim) {}

torch::lazy::NodePtr ScatterReduce::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<ScatterReduce>(operands.at(0), operands.at(1),
                                            operands.at(2), reduce_,
                                            include_self_, dim_);
}

XlaOpVector ScatterReduce::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  xla::XlaOp src = loctx->GetOutputOp(operand(2));
  XlaOpCombiner combiner;
  if (reduce_ == "sum")
    combiner = NumericAddCombiner();
  else if (reduce_ == "prod")
    combiner = NumericMulCombiner();
  else if (reduce_ == "amin")
    combiner = NumericMinCombiner();
  else if (reduce_ == "amax")
    combiner = NumericMaxCombiner();
  else
    XLA_ERROR() << "Reduce type " << reduce_ << " is not supported";
  ScatterOptions options(combiner);
  return ReturnOp(
      CreateScatter(loctx->device(), input, index, src, dim_, options), loctx);
}

std::string ScatterReduce::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  ss << ", reduce=" << reduce_ << ", include_self=" << include_self_;
  return ss.str();
}

}  // namespace torch_xla
