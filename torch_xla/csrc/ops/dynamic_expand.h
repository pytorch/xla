#pragma once

#include "torch_xla/csrc/ir.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class DynamicExpand2 : public Node {
 public:
  DynamicExpand2(Value& lhs, Value& sz);

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
