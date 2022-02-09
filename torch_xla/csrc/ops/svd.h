#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class SVD : public Node {
 public:
  SVD(const Value& input, bool full_matrices, bool compute_uv,
      bool deprecated_svd);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool full_matrices() const { return full_matrices_; }

  bool compute_uv() const { return compute_uv_; }

 private:
  bool full_matrices_;
  bool compute_uv_;
  bool deprecated_svd_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
