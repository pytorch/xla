#include "torch_xla/csrc/ops/scan.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
  xla::Shape input_shape = GetXlaShape(input);
  return input_shape;
}

}  // namespace

Scan::Scan(const Callable f, const at::Tensor& init, const at::Tensor& xs)
    : XlaNode(torch::lazy::OpKind(at::aten::scan), {f, init, xs},
              [&]() { return NodeOutputShape(init); }, 2,) {}

torch::lazy::NodePtr Scan::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Scan>(operands.at(0), operands.at(1), operands.at(2));
}

XlaOpVector Scan::Lower(LoweringContext* loctx) const {
  xla::XlaOp f = loctx->GetOutputOp(operand(0));
  xla::XlaOp init = loctx->GetOutputOp(operand(1));
  xla::XlaOp xs = loctx->GetOutputOp(operand(2));
  return ReturnOps(BuildScan(f, init, xs), loctx);
}

}  // namespace torch_xla
