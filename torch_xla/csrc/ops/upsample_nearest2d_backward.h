#ifndef XLA_TORCH_XLA_CSRC_OPS_UPSAMPLE_NEAREST2D_BACKWARD_H_
#define XLA_TORCH_XLA_CSRC_OPS_UPSAMPLE_NEAREST2D_BACKWARD_H_

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class UpsampleNearestBackward : public XlaNode {
 public:
  UpsampleNearestBackward(const torch::lazy::Value& input,
                          std::vector<int64_t> output_size,
                          std::vector<int64_t> input_size);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

  const std::vector<int64_t>& input_size() const { return input_size_; }

 private:
  std::vector<int64_t> output_size_;
  std::vector<int64_t> input_size_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_UPSAMPLE_NEAREST2D_BACKWARD_H_