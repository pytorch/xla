#include "torch_xla/csrc/ops/as_strided.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <ATen/core/aten_interned_strings.h>
#include <torch/csrc/lazy/core/hash.h>
#include <torch/csrc/lazy/core/ir.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {
namespace {

xla::Shape AsStridedOutputShape(const torch::lazy::Value& input,
                                absl::Span<const int64_t> size) {
  return xla::ShapeUtil::MakeShape(GetXlaShape(input).element_type(), size);
}

}  // namespace

AsStrided::AsStrided(const torch::lazy::Value& input,
                     const std::vector<int64_t>& size,
                     const std::vector<int64_t>& stride, int64_t storage_offset)
    : XlaNode(
          torch::lazy::OpKind(at::aten::as_strided), {input},
          AsStridedOutputShape(input, size),
          /* num_outputs= */ 1,
          /* hash_seed= */ torch::lazy::MHash(size, stride, storage_offset)),
      size_(size),
      stride_(stride),
      storage_offset_(storage_offset) {
  // Make sure `input` has enough elements to fit the given spec.
  XLA_CHECK_OK(CheckSpecFitsInput(input));
}

std::string AsStrided::ToString() const {
  return absl::StrCat(XlaNode::ToString(), ", size=(",
                      absl::StrJoin(size_, ", "), "), stride=(",
                      absl::StrJoin(stride_, ", "),
                      "), storage_offset=", storage_offset_);
}

torch::lazy::NodePtr AsStrided::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<AsStrided>(operands.at(0), size_, stride_,
                                        storage_offset_);
}

absl::StatusOr<XlaOpVector> AsStrided::SafeLower(LoweringContext* loctx) const {
  XLA_ASSIGN_OR_RETURN(xla::XlaOp input, loctx->SafeGetOutputOp(operand(0)));

  XLA_ASSIGN_OR_RETURN(const xla::Shape* absl_nonnull input_shape_ptr,
                       GetShape(input));

  XLA_ASSIGN_OR_RETURN(int64_t input_element_count,
                       xla::ShapeUtil::ElementsIn(*input_shape_ptr));

  int64_t spec_element_count = GetSpecElementCount();

  // Preprocess `input` so that it:
  //   1. Starts from `storage_offset_`
  //   2. Has the same element count as the spec
  //
  // This preprocessing should only be done if:
  //   1. There's actually a `storage_offset_` to start from; or
  //   2. The element count of the input is different from the spec
  if (storage_offset_ > 0 || input_element_count != spec_element_count) {
    XLA_ASSIGN_OR_RETURN(xla::XlaOp flattened, XlaHelpers::SafeFlatten(input));
    input = xla::SliceInDim(flattened, storage_offset_,
                            storage_offset_ + spec_element_count, 1, 0);
  }

  // Since PyTorch/XLA has no concept of strides in a tensor (i.e. all tensors
  // are contiguous), we need a way to compute a contiguous tensor that accesses
  // the same elements that a similarly spec'd strided tensor would. In order to
  // do that, we need to:
  //
  //   1. Reshape the `input`, so that the dimensions with larger strides come
  //      first. This should yield the correct contiguous tensor, but with
  //      permuted dimensions.
  std::vector<int64_t> permutation =
      xla::InversePermutation(GetDescendingOrderPermutation(stride_));
  std::vector<int64_t> permuted_sizes = xla::PermuteInverse(size_, permutation);
  XLA_ASSIGN_OR_RETURN(xla::XlaOp input_reshaped_with_permuted_sizes,
                       XlaHelpers::SafeDynamicReshape(input, permuted_sizes));

  //   2. Reverse the dimension permutation we did on `input` in the previous
  //      step.
  xla::XlaOp output =
      xla::IsIdentityPermutation(permutation)
          ? input_reshaped_with_permuted_sizes
          : xla::Transpose(input_reshaped_with_permuted_sizes, permutation);

  return ReturnOp(output, loctx);
}

int64_t AsStrided::GetSpecElementCount() const {
  return runtime::util::Multiply<int64_t>(size_);
}

absl::Status AsStrided::CheckSpecFitsInput(xla::XlaOp input) const {
  XLA_ASSIGN_OR_RETURN(const xla::Shape* absl_nonnull shape_ptr,
                       GetShape(input));
  XLA_RETURN_IF_ERROR(
      CheckSpecFitsInputImpl(*shape_ptr, xla::ShapeUtil::ElementsIn(*shape_ptr),
                             GetSpecElementCount()));
  return absl::OkStatus();
}

absl::Status AsStrided::CheckSpecFitsInput(
    const torch::lazy::Value& input) const {
  const xla::Shape& shape = GetXlaShape(input);
  XLA_RETURN_IF_ERROR(CheckSpecFitsInputImpl(
      shape, xla::ShapeUtil::ElementsIn(shape), GetSpecElementCount()));
  return absl::OkStatus();
}

absl::Status AsStrided::CheckSpecFitsInputImpl(
    const xla::Shape& input_shape, int64_t input_element_count,
    int64_t spec_element_count) const {
  if (input_element_count < storage_offset_ + spec_element_count) {
    return XLA_ERROR_WITH_LOCATION(absl::InternalError(absl::StrCat(
        "as_strided(): expected input ", input_shape.ToString(),
        " (elements=", input_element_count,
        ") to have enough elements to fit the given spec of size=[",
        absl::StrJoin(size_, /* separator= */ ", "), "], stride=[",
        absl::StrJoin(stride_, /* separator= */ ", "), "], and storage_offset=",
        storage_offset_, " (elements=", spec_element_count, ")")));
  }
  return absl::OkStatus();
}

bool AsStridedIsSupported(const xla::Shape& input_shape,
                          absl::Span<const int64_t> size,
                          absl::Span<const int64_t> stride,
                          int64_t storage_offset) {
  std::vector<int64_t> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<int64_t> GetDescendingOrderPermutation(
    absl::Span<const int64_t> v) {
  std::vector<int64_t> permutation(v.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::sort(permutation.begin(), permutation.end(),
            [&](int64_t a, int64_t b) { return v[a] > v[b]; });
  return permutation;
}

}  // namespace torch_xla
