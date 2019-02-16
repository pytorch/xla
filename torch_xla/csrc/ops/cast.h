#pragma once

#include <c10/core/ScalarType.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Cast : public Node {
 public:
  Cast(const Value& input, at::ScalarType dtype);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  at::ScalarType dtype() const { return dtype_; }

 private:
  at::ScalarType dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
