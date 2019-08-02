#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class NllLossBackward : public Node {
 public:
  NllLossBackward(const Value& logits, const Value& labels, int ignore_index);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int ignore_index() const { return ignore_index_; }

 private:
  int ignore_index_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
