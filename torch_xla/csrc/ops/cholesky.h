#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Cholesky : public Node {
 public:
  Cholesky(const Value& input, bool lower);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool lower() const { return lower_; }

 private:
  bool lower_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
