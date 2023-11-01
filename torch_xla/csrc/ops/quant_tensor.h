#ifndef XLA_TORCH_XLA_CSRC_OPS_QUANT_TENSOR_H_
#define XLA_TORCH_XLA_CSRC_OPS_QUANT_TENSOR_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class QuantizeTensor : public XlaNode {
 public:
  QuantizeTensor(const torch::lazy::Value& input,
                 const std::vector<float>& scale,
                 const std::vector<float>& zero_point, int quant_min,
                 int quant_max, const std::string& dtype, int axis);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int quant_min_;
  int quant_max_;
  int axis_;
  std::string dtype_;
  std::vector<float> scale_;
  std::vector<float> zero_point_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_QUANT_TENSOR_H_