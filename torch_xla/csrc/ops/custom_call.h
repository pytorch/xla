#ifndef XLA_TORCH_XLA_CSRC_OPS_CUSTOM_CALL_H_
#define XLA_TORCH_XLA_CSRC_OPS_CUSTOM_CALL_H_

#include <unordered_map>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class CustomCall : public XlaNode {
 public:
  CustomCall(torch::lazy::OpList inputs, const std::string& call_target,
             xla::Shape output_shape, bool has_side_effect,
             const std::string& backend_config, const int api_version);
  CustomCall(
      torch::lazy::OpList inputs, const std::string& call_target,
      xla::Shape output_shape, bool has_side_effect,
      const std::string& backend_config, const int api_version,
      const std::unordered_map<std::string, std::string>& frontend_attributes);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::string call_target_;
  bool has_side_effect_;
  std::string backend_config_;
  int api_version_;
  std::unordered_map<std::string, std::string> frontend_attributes_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CUSTOM_CALL_H_
