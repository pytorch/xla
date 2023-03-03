#include "torch_xla/csrc/ops/as_strided_view_update.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/as_strided.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerAsStridedViewUpdate(xla::XlaOp target, xla::XlaOp input,
                                    absl::Span<const int64_t> size,
                                    absl::Span<const int64_t> stride,
                                    int64_t storage_offset) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  int64_t input_element_count = xla::ShapeUtil::ElementsIn(input_shape);
  int64_t slice_size = xla::util::Multiply<int64_t>(size);
  XLA_CHECK_LE(storage_offset + input_element_count, slice_size);

  std::vector<int64_t> permutation =
      AsStrided::GetArrayStridePermutation(stride, input_shape.dimensions());
  xla::XlaOp transposed_input = xla::IsIdentityPermutation(permutation)
                                    ? input
                                    : xla::Transpose(input, permutation);
  if (storage_offset > 0 || input_element_count < slice_size) {
    xla::XlaOp r1_input = XlaHelpers::Flatten(transposed_input);
    xla::XlaOp r1_target = XlaHelpers::Flatten(target);
    transposed_input = xla::DynamicUpdateSlice(
        r1_target, r1_input,
        {XlaHelpers::ScalarValue<int64_t>(storage_offset, input.builder())});
  }
  return XlaHelpers::DynamicReshape(transposed_input, size);
}

}  // namespace

AsStridedViewUpdate::AsStridedViewUpdate(const torch::lazy::Value& target,
                                         const torch::lazy::Value& input,
                                         std::vector<int64_t> size,
                                         std::vector<int64_t> stride,
                                         int64_t storage_offset)
    : XlaNode(xla_as_strided_view_update, {target, input},
              [&]() {
                return xla::ShapeUtil::MakeShape(
                    GetXlaShape(target).element_type(), size);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(size, stride, storage_offset)),
      size_(std::move(size)),
      stride_(std::move(stride)),
      storage_offset_(storage_offset) {}

std::string AsStridedViewUpdate::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(size_, ", ")
     << "), stride=(" << absl::StrJoin(stride_, ", ")
     << "), storage_offset=" << storage_offset_;
  return ss.str();
}

torch::lazy::NodePtr AsStridedViewUpdate::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<AsStridedViewUpdate>(
      operands.at(0), operands.at(1), size_, stride_, storage_offset_);
}

XlaOpVector AsStridedViewUpdate::Lower(LoweringContext* loctx) const {
  xla::XlaOp target = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      LowerAsStridedViewUpdate(target, input, size_, stride_, storage_offset_),
      loctx);
}

}  // namespace torch_xla
