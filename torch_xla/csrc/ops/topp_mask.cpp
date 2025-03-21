#include "torch_xla/csrc/ops/topp_mask.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

TopPMask::TopPMask(const torch::lazy::Value& input, float p, int64_t dim,
                   bool stable)
    : XlaNode(torch::lazy::OpKind(at::aten::topk), {input}, GetXlaShape(input),
              /*num_outputs=*/1, torch::lazy::MHash(p, dim, stable)),
      p_(p),
      dim_(dim),
      stable_(stable) {}

torch::lazy::NodePtr TopPMask::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<TopPMask>(operands.at(0), p_, dim_, stable_);
}

XlaOpVector TopPMask::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(CreateTopPMask(loctx->device(), input, p_, dim_, stable_),
                  loctx);
}

std::string TopPMask::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", p=" << p_ << ", dim=" << dim_
     << ", stable=" << stable_;
  return ss.str();
}

}  // namespace torch_xla
