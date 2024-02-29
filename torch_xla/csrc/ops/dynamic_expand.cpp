#include "torch_xla/csrc/ops/dynamic_expand.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const std::vector<int64_t>& size,
                           const torch::lazy::Value& src_tensor, int src_index,
                           int target_index) {
  std::vector<int64_t> expanded_size(size);
  xla::Shape input_shape = GetXlaShape(input);
  std::cout << "check src size: " << input_shape << std::endl;
  for (size_t i = 0; i < expanded_size.size(); ++i) {
    if (expanded_size[i] == -1) {
      expanded_size[i] = input_shape.dimensions(i);
    }
  }
  expanded_size[target_index] = GetXlaShape(src_tensor).dimensions(src_index);
  std::cout << "check expended size: " << expanded_size << std::endl;
  return xla::ShapeUtil::MakeShape(GetXlaShape(input).element_type(),
                                   {expanded_size});
}

}  // namespace

DynamicExpand::DynamicExpand(const torch::lazy::Value& input,
                             const std::vector<int64_t>& size,
                             const torch::lazy::Value& src_tensor,
                             int src_index, int target_index)
    : XlaNode(xla_dynamic_expand, {input, src_tensor},
              NodeOutputShape(
                  input, size, src_tensor, src_index,
                  target_index) /* fix when quant type is added to HLO */,
              /*num_outputs=*/1,
              torch::lazy::MHash(size, src_index, target_index)),
      size_(size),
      src_index_(src_index),
      target_index_(target_index) {}

torch::lazy::NodePtr DynamicExpand::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<DynamicExpand>(
      operands.at(0), size_, operands.at(1), src_index_, target_index_);
}

XlaOpVector DynamicExpand::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp src_tensor = loctx->GetOutputOp(operand(1));

  xla::XlaOp dynamic_dim_tensor =
      xla::Reshape(xla::GetDimensionSize(src_tensor, src_index_), {1});

  // Only support the source index and target index are both 0
  std::vector<int32_t> static_input_dims_vec(size_.begin() + 1, size_.end());
  xla::XlaOp static_input_dims = xla::ConstantR1(
      loctx->builder(), absl::Span<const int32_t>(static_input_dims_vec));
  xla::XlaOp final_broadcast_dimensions = xla::ConcatInDim(
      loctx->builder(), {dynamic_dim_tensor, static_input_dims}, 0);

  // Output shape
  xla::Shape final_shape = ShapeHelper::ShapeOfXlaOp(input);
  final_shape.set_unbounded_dynamic_dimension(target_index_);
  xla::XlaOp result = XlaHelpers::DynamicBroadcastInDim(
      input, final_shape, final_broadcast_dimensions);
  return ReturnOp(result, loctx);
}

std::string DynamicExpand::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=" << size_
     << ", src_index=" << src_index_ << ", target_index=" << target_index_;
  return ss.str();
}

}  // namespace torch_xla
