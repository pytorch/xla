#pragma once

#include <vector>

#include <c10/util/Optional.h>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AsStrided : public Node {
 public:
  AsStrided(const Value& input, std::vector<xla::int64> size,
            c10::optional<xla::int64> storage_offset);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& size() const { return size_; }

  c10::optional<xla::int64> storage_offset() const { return storage_offset_; }

  // We only support strides which are already consistent with the size.
  static bool StrideIsSupported(tensorflow::gtl::ArraySlice<xla::int64> size,
                                tensorflow::gtl::ArraySlice<xla::int64> stride);

 private:
  std::vector<xla::int64> size_;
  c10::optional<xla::int64> storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
