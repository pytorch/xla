#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Permute : public Node {
 public:
  Permute(const Value& input, std::vector<xla::int64_t> dims);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64_t>& dims() const { return dims_; }

  static xla::Shape MakePermuteShape(
      const xla::Shape& source_shape,
      absl::Span<const xla::int64_t> permutation);

 private:
  // The permutation of dimensions.
  std::vector<xla::int64_t> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
