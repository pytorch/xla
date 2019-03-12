#pragma once

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Repeat : public Node {
 public:
  Repeat(const Value& input, std::vector<xla::int64> repeats);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& repeats() const { return repeats_; }

 private:
  // The number of repeats along each dimension.
  std::vector<xla::int64> repeats_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
