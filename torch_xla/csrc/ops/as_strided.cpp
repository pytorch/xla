#include "torch_xla/csrc/ops/as_strided.h"

#include <torch/csrc/lazy/core/util.h>

#include <algorithm>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerAsStrided(xla::XlaOp input, absl::Span<const int64_t> size,
                          absl::Span<const int64_t> stride,
                          int64_t storage_offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  int64_t input_element_count = xla::ShapeUtil::ElementsIn(input_shape);
  int64_t slice_size = xla::util::Multiply<int64_t>(size);
  XLA_CHECK_LE(storage_offset + slice_size, input_element_count);

  xla::XlaOp off_input = input;
  if (storage_offset > 0 || slice_size < input_element_count) {
    xla::XlaOp r1_input = XlaHelpers::Flatten(input);
    off_input = xla::SliceInDim(r1_input, storage_offset,
                                storage_offset + slice_size, 1, 0);
  }

  std::vector<int64_t> permutation = xla::InversePermutation(
      AsStrided::GetArrayStridePermutation(stride, size));
  std::vector<int64_t> new_sizes = xla::PermuteInverse(size, permutation);
  xla::XlaOp reshaped_input = XlaHelpers::DynamicReshape(off_input, new_sizes);
  return xla::IsIdentityPermutation(permutation)
             ? reshaped_input
             : xla::Transpose(reshaped_input, permutation);
}

}  // namespace

AsStrided::AsStrided(const torch::lazy::Value& input, std::vector<int64_t> size,
                     std::vector<int64_t> stride, int64_t storage_offset)
    : XlaNode(torch::lazy::OpKind(at::aten::as_strided), {input},
              [&]() {
                return xla::ShapeUtil::MakeShape(
                    GetXlaShape(input).element_type(), size);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(size_, ", ")
     << "), stride=(" << absl::StrJoin(stride_, ", ")
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

torch::lazy::NodePtr AsStrided::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<AsStrided>(operands.at(0), size_, stride_,
                                          storage_offset_);
}

XlaOpVector AsStrided::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerAsStrided(input, size_, stride_, storage_offset_),
                  loctx);
}

bool AsStrided::StrideIsSupported(const xla::Shape& input_shape,
                                  absl::Span<const int64_t> size,
                                  absl::Span<const int64_t> stride,
                                  int64_t storage_offset) {
  std::vector<int64_t> sorted_stride(stride.begin(), stride.end());
  std::sort(sorted_stride.begin(), sorted_stride.end());
  return stride.empty() || sorted_stride.front() == 1;
}

std::vector<int64_t> AsStrided::GetArrayStridePermutation(
    absl::Span<const int64_t> stride, absl::Span<const int64_t> size) {
  std::vector<int64_t> permutation = torch::lazy::Iota<int64_t>(stride.size());
  std::sort(permutation.begin(), permutation.end(),
            [&](int64_t a, int64_t b) { return stride[a] > stride[b]; });
  return permutation;
}

}  // namespace torch_xla
