#include "torch_xla/csrc/ops/randperm_out.h"

#include "torch_xla/csrc/lowering_context.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

// xw32: should I include a xla::Shape param in the constructor?
// xw32: what opkind should I use?
// xw32: what does the shape param in the constructor mean?
RandpermOut::RandpermOut(const torch::lazy::Value& n, const xla::Shape& shape)
    : XlaNode(torch::lazy::OpKind(at::aten::randperm), {n},
              shape,
              /*num_outputs=*/1, torch::lazy::Hash(shape)) {}

torch::lazy::NodePtr RandpermOut::Clone(
    torch::lazy::OpList operands) const {
        return torch::lazy::MakeNode<RandpermOut>(operands.at(0), xla_shape());
}

XlaOpVector RandpermOut::Lower(LoweringContext* loctx) const {
    xla::XlaOp n = loctx->GetOutputOp(operand(0));
    xla::XlaOp op_randperm_out = BuildRandpermOut(n, loctx->builder());
    return ReturnOp(op_randperm_out, loctx);
}


} // namespace torch_xla