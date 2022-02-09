#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class GenericSlice : public Node {
 public:
  GenericSlice(const Value& input, absl::Span<const int64_t> base_indices,
               absl::Span<const int64_t> sizes);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const { return base_indices_; }

  const std::vector<int64_t>& sizes() const { return sizes_; }

 private:
  std::vector<int64_t> base_indices_;
  std::vector<int64_t> sizes_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
