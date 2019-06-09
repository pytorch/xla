#pragma once

#include <string>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class NotSupported : public Node {
 public:
  NotSupported(std::string description, xla::Shape shape);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::string& description() const { return description_; }

 private:
  std::string description_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
