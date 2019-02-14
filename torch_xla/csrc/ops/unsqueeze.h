#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Unsqueeze : public Node {
 public:
  // Insert a dimension of size one at the specified position.
  Unsqueeze(const Value& input, int dim);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int dim() const { return dim_; }

 private:
  // Position to unsqueeze.
  int dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
