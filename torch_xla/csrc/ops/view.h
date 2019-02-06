#pragma once

#include "ir.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {
namespace ir {
namespace ops {

// IR node for a tensor view.
class View : public Node {
 public:
  View(const Value& input,
       tensorflow::gtl::ArraySlice<const xla::int64> output_size);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<xla::int64>& output_size() const { return output_size_; }

 private:
  // The possibly incomplete output size.
  std::vector<xla::int64> output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
