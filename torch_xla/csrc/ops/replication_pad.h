#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class ReplicationPad : public Node {
 public:
  ReplicationPad(const Value& input, std::vector<xla::int64> padding);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& padding() const { return padding_; }

 private:
  std::vector<xla::int64> padding_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
