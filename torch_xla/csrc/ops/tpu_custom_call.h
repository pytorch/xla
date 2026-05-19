#ifndef XLA_TORCH_XLA_CSRC_OPS_TPU_CUSTOM_CALL_H_
#define XLA_TORCH_XLA_CSRC_OPS_TPU_CUSTOM_CALL_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class TpuCustomCall : public XlaNode {
 public:
  // Make a TPU custom call with payload, e.g., Mosaic.
  TpuCustomCall(torch::lazy::OpList inputs, xla::Shape output_shape,
                const std::string& payload);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

 private:
  std::string payload_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_TPU_CUSTOM_CALL_H_
