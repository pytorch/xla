#pragma once

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AsStrided : public Node {
 public:
  AsStrided(const Value& input, std::vector<xla::int64_t> size,
            std::vector<xla::int64_t> stride, xla::int64_t storage_offset);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64_t>& size() const { return size_; }

  const std::vector<xla::int64_t>& stride() const { return stride_; }

  xla::int64_t storage_offset() const { return storage_offset_; }

  static bool StrideIsSupported(const xla::Shape& input_shape,
                                absl::Span<const xla::int64_t> size,
                                absl::Span<const xla::int64_t> stride,
                                xla::int64_t storage_offset);

  static std::vector<xla::int64_t> GetArrayStridePermutation(
      absl::Span<const xla::int64_t> stride, absl::Span<const xla::int64_t> size);

 private:
  std::vector<xla::int64_t> size_;
  std::vector<xla::int64_t> stride_;
  xla::int64_t storage_offset_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
