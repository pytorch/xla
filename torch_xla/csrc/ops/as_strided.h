#ifndef XLA_TORCH_XLA_CSRC_OPS_AS_STRIDED_H_
#define XLA_TORCH_XLA_CSRC_OPS_AS_STRIDED_H_

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class AsStrided : public XlaNode {
 public:
  AsStrided(const torch::lazy::Value& input, std::vector<int64_t> size,
            std::vector<int64_t> stride, int64_t storage_offset);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& size() const { return size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  int64_t storage_offset() const { return storage_offset_; }

  static bool StrideIsSupported(const xla::Shape& input_shape,
                                absl::Span<const int64_t> size,
                                absl::Span<const int64_t> stride,
                                int64_t storage_offset);

  static std::vector<int64_t> GetArrayStridePermutation(
      absl::Span<const int64_t> stride, absl::Span<const int64_t> size);

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_AS_STRIDED_H_