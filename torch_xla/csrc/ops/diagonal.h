#ifndef XLA_TORCH_XLA_CSRC_OPS_DIAGONAL_H_
#define XLA_TORCH_XLA_CSRC_OPS_DIAGONAL_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Diagonal : public XlaNode {
 public:
  Diagonal(const torch::lazy::Value& input, int64_t offset, int64_t dim1,
           int64_t dim2);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t offset() const { return offset_; }

  int64_t dim1() const { return dim1_; }

  int64_t dim2() const { return dim2_; }

  static xla::Shape MakeDiagonalShape(const xla::Shape& shape, int64_t offset,
                                      int64_t dim1, int64_t dim2);

 private:
  int64_t offset_;
  int64_t dim1_;
  int64_t dim2_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_DIAGONAL_H_