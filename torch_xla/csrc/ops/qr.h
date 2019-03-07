#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class QR : public Node {
 public:
  QR(const Value& input, bool full_matrices);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool full_matrices() const { return full_matrices_; }

 private:
  bool full_matrices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
