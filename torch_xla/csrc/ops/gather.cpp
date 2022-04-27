#include "torch_xla/csrc/ops/gather.h"

#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {

Gather::Gather(const XlaValue& input, int64_t dim, const XlaValue& index)
    : XlaNode(torch::lazy::OpKind(at::aten::gather), {input, index},
              xla::ShapeUtil::MakeShape(input.xla_shape().element_type(),
                                        index.xla_shape().dimensions()),
              /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr Gather::Clone(OpList operands) const {
  return torch::lazy::MakeNode<Gather>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector Gather::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::TorchGather(input, index, dim_, /*sparse=*/true), loctx);
}

std::string Gather::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

} // namespace torch_xla
