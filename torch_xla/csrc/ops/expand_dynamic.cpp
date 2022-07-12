#include "torch_xla/csrc/ops/expand_dynamic.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch/csrc/lazy/core/helpers.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

std::vector<xla::Shape> GetShapes(const torch::lazy::Value& input,
                                  const std::vector<int64_t> upper_bounds,
                                  const std::vector<bool> dynamic_dims) {
  std::vector<xla::Shape> shapes;
  shapes.push_back(GetXlaShape(input));
  for (int i = 0; i < upper_bounds.size(); i++) {
    shapes.push_back(
        xla::ShapeUtil::MakeShape(GetXlaShape(input).element_type(),
                                  {upper_bounds[i]}, {dynamic_dims[i]}));
  }
  return shapes;
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const std::vector<int64_t> upper_bounds,
                           const std::vector<bool> dynamic_dims) {
  std::vector<xla::Shape> shapes = GetShapes(input, upper_bounds, dynamic_dims);
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_GE(operands.size(), 2) << operands.size();
    return SetDimensionSizes(BuildExpand(operands[0], upper_bounds),
                             operands.subspan(1));
  };
  return InferOutputShape(shapes, lower_for_shape_fn);
}

std::vector<torch::lazy::Value> GetValues(
    const torch::lazy::Value& input,
    const std::vector<torch::lazy::Value> dimensions) {
  std::vector<torch::lazy::Value> values = dimensions;
  values.insert(values.begin(), input);
  return values;
}

}  // namespace

ExpandDynamic::ExpandDynamic(const torch::lazy::Value& input,
                             const std::vector<torch::lazy::Value>& dimensions,
                             const std::vector<int64_t> upper_bounds,
                             const std::vector<bool> dynamic_dims)
    : XlaNode(
          torch::lazy::OpKind(at::aten::expand), GetValues(input, dimensions),
          [&]() { return NodeOutputShape(input, upper_bounds, dynamic_dims); },
          /*num_outputs=*/1, torch::lazy::MHash(upper_bounds)),
      upper_bounds_(std::move(upper_bounds)),
      dynamic_dims_(std::move(dynamic_dims)) {
  shapes_ = GetShapes(input, upper_bounds_, dynamic_dims_);
}

XlaOpVector ExpandDynamic::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  std::vector<xla::XlaOp> size_ops;
  for (int i = 1; i < shapes_.size(); i++) {
    xla::XlaOp size_op = xla::Zeros(loctx->builder(), shapes_[i]);
    size_ops.push_back(size_op);
  }
  xla::XlaOp output =
      SetDimensionSizes(BuildExpand(input, upper_bounds_), size_ops);
  return ReturnOp(output, loctx);
}

std::string ExpandDynamic::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(upper_bounds_, ", ")
     << ")"
     << ", dynamic_dims=(" << absl::StrJoin(dynamic_dims_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
