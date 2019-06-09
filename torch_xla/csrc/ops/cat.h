#pragma once

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Cat : public Node {
 public:
  Cat(tensorflow::gtl::ArraySlice<const ir::Value> values, xla::int64 dim);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  xla::int64 dim() const { return dim_; };

 private:
  xla::int64 dim_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
