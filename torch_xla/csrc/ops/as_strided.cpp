#include "torch_xla/csrc/ops/as_strided.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerAsStrided(xla::XlaOp input, absl::Span<const xla::int64> size,
                          xla::int64 storage_offset) {
  xla::int64 input_element_count =
      xla::ShapeUtil::ElementsIn(XlaHelpers::ShapeOfXlaOp(input));
  xla::int64 slice_size = xla::util::Multiply<xla::int64>(size);
  if (input_element_count == slice_size) {
    XLA_CHECK_EQ(storage_offset, 0);
    return XlaHelpers::DynamicReshape(input, size);
  }
  xla::XlaOp r1_slice =
      XlaHelpers::DynamicReshape(input, {input_element_count});
  xla::XlaOp r1_result = xla::SliceInDim(r1_slice, storage_offset,
                                         storage_offset + slice_size, 1, 0);
  return xla::Reshape(r1_result, size);
}

xla::Shape NodeOutputShape(const Value& input,
                           absl::Span<const xla::int64> size,
                           xla::int64 storage_offset) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerAsStrided(operands[0], size, storage_offset);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

AsStrided::AsStrided(const Value& input, std::vector<xla::int64> size,
                     xla::int64 storage_offset)
    : Node(ir::OpKind(at::aten::as_strided), {input},
           [&]() { return NodeOutputShape(input, size, storage_offset); },
           /*num_outputs=*/1, xla::util::MHash(size, storage_offset)),
      size_(std::move(size)),
      storage_offset_(storage_offset) {}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=[" << absl::StrJoin(size_, ", ")
     << "], storage_offset=" << storage_offset_;
  return ss.str();
}

NodePtr AsStrided::Clone(OpList operands) const {
  return MakeNode<AsStrided>(operands.at(0), size_, storage_offset_);
}

XlaOpVector AsStrided::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerAsStrided(input, size_, storage_offset_), loctx);
}

bool AsStrided::StrideIsSupported(absl::Span<const xla::int64> size,
                                  absl::Span<const xla::int64> stride) {
  XLA_CHECK_EQ(size.size(), stride.size());
  std::vector<xla::int64> expected_stride(size.size(), 1);
  for (size_t i = size.size(); i > 1; --i) {
    expected_stride[i - 2] = expected_stride[i - 1] * size[i - 1];
  }
  return stride == expected_stride;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
