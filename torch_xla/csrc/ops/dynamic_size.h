#pragma once

#include "torch_xla/csrc/ir.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_shape_inference.h"
#include "lazy_tensor_core/csrc/ts_backend/ts_node_lowering.h"

namespace torch_xla {
namespace ir {
namespace ops {

class DynamicSize2 : public Node {
 public:
  DynamicSize2(Value lhs);

  XlaOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function, //TODO: milad fix this
                   LoweringContext* loctx) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
