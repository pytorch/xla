#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class UpsampleNearest : public Node {
 public:
  UpsampleNearest(const Value& input, std::vector<xla::int64> output_size);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& output_size() const { return output_size_; }

 private:
  std::vector<xla::int64> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
