#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Cast : public Node {
 public:
  Cast(const Value& input, xla::PrimitiveType type);
  Cast(const Value& input, at::ScalarType dtype,
       c10::optional<at::ScalarType> stype = c10::nullopt);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::PrimitiveType type() const { return type_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; };

  const c10::optional<at::ScalarType>& stype() const { return stype_; };

 private:
  xla::PrimitiveType type_;
  c10::optional<at::ScalarType> dtype_;
  c10::optional<at::ScalarType> stype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
