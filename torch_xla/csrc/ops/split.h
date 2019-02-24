#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Split the tensor into chunks along a given dimension.
class Split : public Node {
 public:
  Split(const Value& input, std::vector<xla::int64> split_sizes, xla::int64 dim);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& split_sizes() const { return split_sizes_; }

  xla::int64 dim() const { return dim_; }

 private:
  std::vector<xla::int64> split_sizes_;
  xla::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
