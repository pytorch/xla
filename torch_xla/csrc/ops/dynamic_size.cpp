#include "torch_xla/csrc/ops/dynamic_size.h"


namespace torch_xla {
namespace ir {
namespace ops {

DynamicSize2::DynamicSize2(Value lhs)
    : Node(OpKind(c10::Symbol::prim("_dynamic_size2")), lhs,
             {ir::GetShapeFromTsValue(lhs)}) {}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
