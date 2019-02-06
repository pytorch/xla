#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ThresholdBackward : public Node {
 public:
  ThresholdBackward(const Value& grad_output, const Value& input,
                    float threshold);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

 private:
  float threshold_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
