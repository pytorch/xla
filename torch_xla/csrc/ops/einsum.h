#pragma once

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Einsum : public Node {
 public:
  Einsum(const std::string& equation,
         tensorflow::gtl::ArraySlice<const ir::Value> values);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::string& equation() const { return equation_; }

  static bool SupportsEquation(const std::string& equation, xla::int64 x_rank,
                               xla::int64 y_rank);

 private:
  std::string equation_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
