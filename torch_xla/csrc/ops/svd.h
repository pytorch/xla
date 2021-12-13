#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class SVD : public XlaNode {
 public:
  SVD(const XlaValue& input, bool some, bool compute_uv);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool some() const { return some_; }

  bool compute_uv() const { return compute_uv_; }

 private:
  bool some_;
  bool compute_uv_;
};

class LinalgSVD : public Node {
 public:
  LinalgSVD(const Value& input, bool full_matrices);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool full_matrices() const { return full_matrices_; }

 private:
  bool full_matrices_;
};

}  // namespace torch_xla
