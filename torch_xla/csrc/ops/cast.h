#ifndef XLA_TORCH_XLA_CSRC_OPS_CAST_H_
#define XLA_TORCH_XLA_CSRC_OPS_CAST_H_

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Cast : public XlaNode {
 public:
  Cast(const torch::lazy::Value& input, xla::PrimitiveType type);
  Cast(const torch::lazy::Value& input, at::ScalarType dtype,
       c10::optional<at::ScalarType> stype = c10::nullopt);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::PrimitiveType type() const { return type_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; };

  const c10::optional<at::ScalarType>& stype() const { return stype_; };

 private:
  xla::PrimitiveType type_;
  c10::optional<at::ScalarType> dtype_;
  c10::optional<at::ScalarType> stype_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_CAST_H_