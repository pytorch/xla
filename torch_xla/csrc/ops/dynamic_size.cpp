#include "torch_xla/csrc/ops/dynamic_size.h"


namespace torch_xla {
namespace ir {
namespace ops {

DynamicSize2::DynamicSize2(Value lhs)
    : Node(OpKind(c10::Symbol::prim("_dynamic_size2")), lhs,
             {ir::GetShapeFromTsValue(lhs)}) {}

XlaOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function, //TODO: milad fix this
                 LoweringContext* loctx) const override {

  CHECK(operands().size() == 1);
  auto graph = function->graph();

  auto size_val = graph->insert(at::aten::size, {loctx->GetOutputOp(operands().at(0))});
  return {size_val};
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
