#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Nms : public Node {
 public:
  Nms(const Value& boxes, const Value& scores, const Value& score_threshold,
      const Value& iou_threshold, xla::int64 output_size);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 output_size() const { return output_size_; }

 private:
  xla::int64 output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
