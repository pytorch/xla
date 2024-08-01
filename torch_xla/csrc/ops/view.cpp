#include "torch_xla/csrc/ops/view.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           absl::Span<const int64_t> output_sizes) {
  const xla::Shape& input_shape = GetXlaShape(input);
  auto info = XlaHelpers::GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return std::move(info->output_shape);
  }
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, input_shape.dimensions());
  return xla::ShapeUtil::MakeShape(input_shape.element_type(),
                                   complete_output_sizes);
}

}  // namespace

ViewOp::ViewOp(const torch::lazy::Value& input,
               std::vector<int64_t> output_size)
    : XlaNode(torch::lazy::OpKind(at::aten::view), {input},
              NodeOutputShape(input, output_size),
              /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

ViewOp::ViewOp(const torch::lazy::Value& input, xla::Shape output_shape)
    : XlaNode(
          torch::lazy::OpKind(at::aten::view), {input}, output_shape,
          /*num_outputs=*/1,
          torch::lazy::MHash(
              torch::lazy::ToVector<int64_t>(output_shape.dimensions()),
              torch::lazy::ToVector<bool>(output_shape.dynamic_dimensions()))),
      output_size_(torch::lazy::ToVector<int64_t>(output_shape.dimensions())) {}

XlaOpVector ViewOp::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output;
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  if (!input_shape.is_unbounded_dynamic()) {
    output = BuildView(input, output_size_);
  } else {
    output = BuildUnboundedDynamicView(input, input_shape, output_size_);
  }
  return ReturnOp(output, loctx);
}

std::string ViewOp::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
