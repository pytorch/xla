#ifndef XLA_TORCH_XLA_CSRC_OPS_DIAGONAL_VIEW_UPDATE_H_
#define XLA_TORCH_XLA_CSRC_OPS_DIAGONAL_VIEW_UPDATE_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class DiagonalViewUpdate : public XlaNode {
 public:
  DiagonalViewUpdate(const torch::lazy::Value& target,
                     const torch::lazy::Value& input, int64_t offset,
                     int64_t dim1, int64_t dim2);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

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

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_DIAGONAL_VIEW_UPDATE_H_