#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class SVD : public Node {
 public:
  SVD(const Value& input, bool some, bool compute_uv);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool some() const { return some_; }

  bool compute_uv() const { return compute_uv_; }

 private:
  bool some_;
  bool compute_uv_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
