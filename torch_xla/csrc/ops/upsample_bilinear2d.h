#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class UpsampleBilinear : public Node {
 public:
  UpsampleBilinear(const Value& input, std::vector<xla::int64> output_size,
                   bool align_corners);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& output_size() const { return output_size_; }

  bool align_corners() const { return align_corners_; }

 private:
  std::vector<xla::int64> output_size_;
  bool align_corners_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
