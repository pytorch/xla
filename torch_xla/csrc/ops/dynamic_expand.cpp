#include "torch_xla/csrc/ops/dynamic_expand.h"


namespace torch_xla {
namespace ir {
namespace ops {

DynamicExpand2::DynamicExpand2(Value lhs, Value sz)
    : Node(OpKind(c10::Symbol::prim("_dynamic_expand2")), {lhs, sz},
             {ir::GetShapeFromTsValue(sz)}) 
             {} //TODO: Milad to resolve the issue

XlaOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                 LoweringContext* loctx) const override {

  CHECK(operands().size() == 2);
  auto graph = function->graph();
  auto sz_val = loctx->GetOutputOp(operand(1));
  auto expand = graph->insert(at::aten::expand, {loctx->GetOutputOp(operands().at(0)), sz_val});
  return {expand};
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
