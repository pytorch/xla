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

xla::XlaOp LowerAsStrided(const xla::XlaOp& input,
                          tensorflow::gtl::ArraySlice<xla::int64> size,
                          c10::optional<xla::int64> storage_offset) {
  xla::int64 input_element_count =
      xla::ShapeUtil::ElementsIn(XlaHelpers::ShapeOfXlaOp(input));
  xla::int64 slice_size = xla::util::Multiply<xla::int64>(size);
  if (input_element_count == slice_size) {
    XLA_CHECK(!storage_offset || *storage_offset == 0)
        << "For same number of elements, storage offset must be 0 or null";
    return xla::Reshape(input, size);
  }
  xla::XlaOp r1_slice = xla::Reshape(input, {input_element_count});
  if (storage_offset) {
    r1_slice = xla::SliceInDim(r1_slice, *storage_offset,
                               *storage_offset + slice_size, 1, 0);
  }
  return xla::Reshape(r1_slice, size);
}

xla::Shape NodeOutputShape(const Value& input,
                           tensorflow::gtl::ArraySlice<xla::int64> size,
                           c10::optional<xla::int64> storage_offset) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return LowerAsStrided(operands[0], size, storage_offset);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

AsStrided::AsStrided(const Value& input, std::vector<xla::int64> size,
                     c10::optional<xla::int64> storage_offset)
    : Node(
          ir::OpKind(at::aten::as_strided), {input},
          [&]() { return NodeOutputShape(input, size, storage_offset); },
          /*num_outputs=*/1,
          xla::util::MHash(size,
                           OptionalOr<xla::int64>(storage_offset, 0x311bd6))),
      size_(std::move(size)),
      storage_offset_(storage_offset) {}

std::string AsStrided::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=[" << absl::StrJoin(size_, ", ")
     << "], storage_offset="
     << (storage_offset_ ? std::to_string(*storage_offset_) : "null");
  return ss.str();
}

XlaOpVector AsStrided::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerAsStrided(input, size_, storage_offset_), loctx);
}

bool AsStrided::StrideIsSupported(
    tensorflow::gtl::ArraySlice<xla::int64> size,
    tensorflow::gtl::ArraySlice<xla::int64> stride) {
  XLA_CHECK_EQ(size.size(), stride.size());
  XLA_CHECK(!size.empty()) << "Output size cannot be empty";
  std::vector<xla::int64> expected_stride(size.size(), 1);
  for (size_t i = size.size() - 1; i > 0; --i) {
    expected_stride[i - 1] = expected_stride[i] * size[i];
  }
  return stride == expected_stride;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
