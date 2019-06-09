#pragma once

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class UpdateSlice : public Node {
 public:
  UpdateSlice(const Value& input, const Value& source,
              tensorflow::gtl::ArraySlice<const xla::int64> base_indices);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& base_indices() const { return base_indices_; }

 private:
  std::vector<xla::int64> base_indices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
