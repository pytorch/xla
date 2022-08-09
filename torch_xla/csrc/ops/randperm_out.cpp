#include "torch_xla/csrc/ops/randperm_out.h"

#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

// xw32: the shape param in the constructor mean the expected output shape.
// xw32: what opkind should I use? randperm or randperm_out? randperm_out
// doesn't seem to exist. xw32: what does the hash in the constructor mean?
RandpermOut::RandpermOut(int64_t n)
    : XlaNode(torch::lazy::OpKind(at::aten::randperm), torch::lazy::OpList(),
              xla::ShapeUtil::MakeShape(
                  xla::S64, {n}),  // TODO: should I use U64 or U32 or others?
                                   // How can I determine that?
              /*num_outputs=*/1, torch::lazy::MHash(n)),
      n_(n) {
  std::cout << "xw32 inside randperm_out.cpp constructor" << std::endl;
}

torch::lazy::NodePtr RandpermOut::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<RandpermOut>(n_);
}

XlaOpVector RandpermOut::Lower(LoweringContext* loctx) const {
  std::cout << "xw32 inside randperm_out.cpp RandpermOut::Lower" << std::endl;
  xla::XlaOp op_randperm_out = BuildRandpermOut(n_, loctx->builder());
  return ReturnOp(op_randperm_out, loctx);
}

std::string RandpermOut::ToString() const {
  std::cout << "xw32 inside randperm_out.cpp RandpermOut::ToString"
            << std::endl;
  std::stringstream ss;
  ss << XlaNode::ToString() << ", n=(" << n_ << ")";
  return ss.str();
}

}  // namespace torch_xla