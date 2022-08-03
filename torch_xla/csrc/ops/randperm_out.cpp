#include "torch_xla/csrc/ops/randperm_out.h"

#include "torch_xla/csrc/lowering_context.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

// xw32: should I include a xla::Shape param in the constructor?
// xw32: what opkind should I use? randperm or randperm_out? randperm_out doesn't seem to exist.
// xw32: what does the shape param in the constructor mean? The output shape?
// xw32: what does the hash in the constructor mean?
RandpermOut::RandpermOut(int64_t n)
    : XlaNode(torch::lazy::OpKind(at::aten::randperm), torch::lazy::OpList(),
              xla::ShapeUtil::MakeShape(xla::U64, {n}),
              /*num_outputs=*/1, torch::lazy::MHash(n)),
              n_(n) {}

torch::lazy::NodePtr RandpermOut::Clone(
    torch::lazy::OpList operands) const {
        return torch::lazy::MakeNode<RandpermOut>(n_);
}

XlaOpVector RandpermOut::Lower(LoweringContext* loctx) const {
    xla::XlaOp op_randperm_out = BuildRandpermOut(n_, loctx->builder());
    return ReturnOp(op_randperm_out, loctx);
}

std::string RandpermOut::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", n=(" << n_ << ")";
  return ss.str();
}

} // namespace torch_xla