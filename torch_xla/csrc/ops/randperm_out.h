#pragma once

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class RandpermOut : public Node {
 public:
  RandpermOut(int64_t n);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t n_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla