#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Split the tensor into chunks along a given dimension.
class Split : public Node {
 public:
  Split(const Value& input, xla::int64 split_size, xla::int64 dim);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 split_size() const { return split_size_; }

  xla::int64 dim() const { return dim_; }

 private:
  xla::int64 split_size_;
  xla::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
