#pragma once

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class GenericSlice : public Node {
 public:
  GenericSlice(const Value& input,
               tensorflow::gtl::ArraySlice<const xla::int64> base_indices,
               tensorflow::gtl::ArraySlice<const xla::int64> sizes);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& base_indices() const { return base_indices_; }

  const std::vector<xla::int64>& sizes() const { return sizes_; }

 private:
  std::vector<xla::int64> base_indices_;
  std::vector<xla::int64> sizes_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
