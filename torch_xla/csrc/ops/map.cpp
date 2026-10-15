#include "torch_xla/csrc/ops/map.h"

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

Map::Map(const Callable f, const at::Tensor& xs)
    : XlaNode(torch::lazy::OpKind(at::aten::map), {f, xs},
              [&]() { return NodeOutputShape(xs); }, 2,) {}

torch::lazy::NodePtr Map::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Map>(operands.at(0), operands.at(1));
}

XlaOpVector Map::Lower(LoweringContext* loctx) const {
  xla::XlaOp f = loctx->GetOutputOp(operand(0));
  xla::XlaOp xs = loctx->GetOutputOp(operand(1));
  return ReturnOps(BuildMap(f, xs), loctx);
}

}  // namespace torch_xla
