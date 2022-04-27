#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class DiagonalViewUpdate : public XlaNode {
 public:
  DiagonalViewUpdate(const XlaValue& target, const XlaValue& input,
                     int64_t offset, int64_t dim1, int64_t dim2);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t offset() const { return offset_; }

  int64_t dim1() const { return dim1_; }

  int64_t dim2() const { return dim2_; }

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

} // namespace torch_xla
