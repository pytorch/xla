#pragma once

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AsStrided : public Node {
 public:
  AsStrided(const Value& input, std::vector<xla::int64> size,
            std::vector<xla::int64> stride, xla::int64 storage_offset);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& size() const { return size_; }

  const std::vector<xla::int64>& stride() const { return stride_; }

  xla::int64 storage_offset() const { return storage_offset_; }

  static bool StrideIsSupported(const xla::Shape& input_shape,
                                absl::Span<const xla::int64> size,
                                absl::Span<const xla::int64> stride,
                                xla::int64 storage_offset);

  static std::vector<xla::int64> GetSliceBaseIndices(
      absl::Span<const xla::int64> stride, xla::int64 storage_offset);

  static std::vector<xla::int64> GetArrayStridePermutation(
      absl::Span<const xla::int64> stride, absl::Span<const xla::int64> size);

 private:
  std::vector<xla::int64> size_;
  std::vector<xla::int64> stride_;
  xla::int64 storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
