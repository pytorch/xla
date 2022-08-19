#include "torch_xla/csrc/ops/expand_symint.h"

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

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const std::vector<int64_t> upper_bounds,
                           const std::vector<bool> dynamic_dims) {
  return xla::ShapeUtil::MakeShape(GetXlaShape(input).element_type(),
                                   {upper_bounds}, {dynamic_dims});
}

std::vector<torch::lazy::Value> GetValues(
    const torch::lazy::Value& input,
    const std::vector<torch::lazy::Value>& dimensions) {
  std::vector<torch::lazy::Value> values = dimensions;
  values.insert(values.begin(), input);
  return values;
}

}  // namespace

ExpandSymInt::ExpandSymInt(const torch::lazy::Value& input, 
                           const SymIntElements& size_elements, 
                           const torch::lazy::Shape& shape)
    : XlaNode(
          torch::lazy::OpKind(at::aten::expand), GetValues(input, size_elements.GetValues()),
          {shape},
          [&]() { return NodeOutputShape(input, size_elements.GetUpperBounds(), size_elements.GetDynamicDims()); },
          /*num_outputs=*/1, torch::lazy::MHash(size_elements.GetUpperBounds(), size_elements.GetDynamicDims())),
      upper_bounds_(std::move(size_elements.GetUpperBounds())),
      dynamic_dims_(std::move(size_elements.GetDynamicDims())) {
}

XlaOpVector ExpandSymInt::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  std::vector<xla::XlaOp> size_ops;
  for (int i = 1; i < operands().size(); i++) {
    size_ops.push_back(loctx->GetOutputOp(operand(i)));
  }
  xla::XlaOp output = SetDimensionSizes(BuildExpand(input, upper_bounds_), size_ops);
  return ReturnOp(output, loctx);
}

std::string ExpandSymInt::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(upper_bounds_, ", ")
     << ")"
     << ", dynamic_dims=(" << absl::StrJoin(dynamic_dims_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
