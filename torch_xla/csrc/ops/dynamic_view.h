#ifndef XLA_TORCH_XLA_CSRC_OPS_DYNAMIC_VIEW_H_
#define XLA_TORCH_XLA_CSRC_OPS_DYNAMIC_VIEW_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class DynamicView : public XlaNode {
 public:
  DynamicView(const torch::lazy::Value& input, const std::vector<int64_t>& size,
              const torch::lazy::Value& src_tensor, int src_index,
              int target_index, float mul_scaler);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::vector<int64_t> size_;
  int src_index_;
  int target_index_;
  float mul_scaler_;
  xla::Shape complete_output_shape_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_DYNAMIC_VIEW_H_
