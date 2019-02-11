#pragma once

#include "ir.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Select : public Node {
 public:
  Select(const Value& input, xla::int64 dim, xla::int64 index);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  xla::int64 dim() const { return dim_; }

  xla::int64 index() const { return index_; }

 private:
  xla::int64 dim_ = 0;
  xla::int64 index_ = 0;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
