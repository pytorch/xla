#ifndef XLA_TORCH_XLA_CSRC_OPS_AS_STRIDED_H_
#define XLA_TORCH_XLA_CSRC_OPS_AS_STRIDED_H_

#include <cstdint>
#include <string>
#include <vector>

#include <torch/csrc/lazy/core/ir.h>

#include "absl/status/statusor.h"
#include "absl/types/span.h"

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

class AsStrided : public XlaNode {
 public:
  AsStrided(const torch::lazy::Value& input, const std::vector<int64_t>& size,
            const std::vector<int64_t>& stride, int64_t storage_offset);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  absl::StatusOr<XlaOpVector> SafeLower(LoweringContext* loctx) const override;

 private:
  std::vector<int64_t> size_;
  std::vector<int64_t> stride_;
  int64_t storage_offset_;

  // Convenient function for retrieving the number of elements given by `size_`.
  int64_t GetSpecElementCount() const;
  // Check that the given as_strided arguments (i.e. spec) are actually within
  // the input tensor bounds. i.e. whether it's actually possible to retrieve a
  // non-overlapping tensor given by spec from the input tensor.
  absl::Status CheckSpecFitsInputImpl(const xla::Shape& input_shape,
                                      int64_t input_element_count,
                                      int64_t spec_element_count) const;
  // Lowering check that calls `CheckSpecFitsInputImpl()`.
  absl::Status CheckSpecFitsInput(xla::XlaOp input) const;
  // Tracing check that calls `CheckSpecFitsInputImpl()`.
  absl::Status CheckSpecFitsInput(const torch::lazy::Value& input) const;
};

// Legacy function that checks whether the given specs are supported by this
// lowering of the `as_strided` operation.
bool IsAsStridedSupported(const xla::Shape& input_shape,
                          absl::Span<const int64_t> size,
                          absl::Span<const int64_t> stride,
                          int64_t storage_offset);

// Retrieves the permutation for sorting `v` in descending order.
std::vector<int64_t> GetDescendingOrderPermutation(absl::Span<const int64_t> v);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_AS_STRIDED_H_
