#include "torch_xla/csrc/ops/dynamic_expand.h"


namespace torch_xla {
namespace ir {
namespace ops {

DynamicExpand2::DynamicExpand2(Value lhs, Value sz)
    : TsNode(OpKind(c10::Symbol::prim("_dynamic_expand2")), {lhs, sz},
             {ir::GetShapeFromTsValue(sz)}) {} //TODO: Milad to resolve the issue


}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors