#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Dropout : public Node {
 public:
  Dropout(const Value& input, double probability);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  double probability() const { return probability_; }

 private:
  double probability_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
