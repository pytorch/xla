#pragma once

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Permute : public Node {
 public:
  Permute(const Value& input, std::vector<xla::int64> dims);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& dims() const { return dims_; }

 private:
  // The permutation of dimensions.
  std::vector<xla::int64> dims_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
