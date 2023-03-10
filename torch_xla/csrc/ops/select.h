#ifndef XLA_TORCH_XLA_CSRC_OPS_SELECT_H_
#define XLA_TORCH_XLA_CSRC_OPS_SELECT_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Select : public XlaNode {
 public:
  Select(const torch::lazy::Value& input, int64_t dim, int64_t start,
         int64_t end, int64_t stride);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

  int64_t start() const { return start_; }

  int64_t end() const { return end_; }

  int64_t stride() const { return stride_; }

  static xla::Shape MakeSelectShape(const xla::Shape& shape, int64_t dim,
                                    int64_t start, int64_t end, int64_t stride);

  static int64_t GetStride(int64_t start, int64_t end, int64_t stride);

 private:
  int64_t dim_;
  int64_t start_;
  int64_t end_;
  int64_t stride_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SELECT_H_