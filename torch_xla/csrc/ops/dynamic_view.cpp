#include "torch_xla/csrc/ops/dynamic_view.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"

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

DynamicView::DynamicView(const torch::lazy::Value& input,
                         const std::vector<int64_t>& size,
                         const torch::lazy::Value& src_tensor, int src_index,
                         int target_index, float mul_scaler)
    : XlaNode(xla_dynamic_view, {input, src_tensor},
              NodeOutputShape(input, size),
              /*num_outputs=*/1,
              torch::lazy::MHash(size, src_index, target_index, mul_scaler)),
      size_(size),
      src_index_(src_index),
      target_index_(target_index),
      mul_scaler_(mul_scaler),
      complete_output_shape_(NodeOutputShape(input, size)) {}

torch::lazy::NodePtr DynamicView::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<DynamicView>(operands.at(0), size_,
                                            operands.at(1), src_index_,
                                            target_index_, mul_scaler_);
}

XlaOpVector DynamicView::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp src_tensor = loctx->GetOutputOp(operand(1));

  xla::XlaOp dynamic_dim_tensor =
      xla::Reshape(xla::GetDimensionSize(src_tensor, src_index_), {1});
  xla::XlaOp dynamic_dim_scaler = xla::ConstantR1(
      loctx->builder(),
      absl::Span<const int32_t>({static_cast<int>(mul_scaler_)}));
  xla::XlaOp dynamic_dim_scaled =
      xla::Mul(dynamic_dim_tensor, dynamic_dim_scaler);

  // Only support the source index and target index are both 0
  std::vector<int32_t> static_input_dims_vec(size_.begin() + 1, size_.end());
  xla::XlaOp static_input_dims = xla::ConstantR1(
      loctx->builder(), absl::Span<const int32_t>(static_input_dims_vec));
  xla::XlaOp final_broadcast_dimensions = xla::ConcatInDim(
      loctx->builder(), {dynamic_dim_scaled, static_input_dims}, 0);

  // Output shape
  xla::Shape final_shape = complete_output_shape_;
  final_shape.set_unbounded_dynamic_dimension(target_index_);
  xla::XlaOp result =
      xla::CustomCall(input.builder(), "mhlo.dynamic_reshape",
                      {input, final_broadcast_dimensions}, final_shape);
  std::cout << ToString() << std::endl;
  std::cout << "check final shape" << final_shape << std::endl;
  return ReturnOp(result, loctx);
}

std::string DynamicView::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=" << size_
     << ", src_index=" << src_index_ << ", target_index=" << target_index_
     << ", mul_scaler=" << mul_scaler_;
  return ss.str();
}

}  // namespace torch_xla
