#include "torch_xla/csrc/ops/unique2.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/convolution_overrideable.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input, bool return_inverse,
                           bool return_counts) {
  xla::Shape input_shape = GetXlaShape(input);
  int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::Shape unique_elements_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), {num_elements});
  xla::Shape inverse_indices_shape =
      xla::ShapeUtil::MakeShape(xla::S64, input_shape.dimensions());
  xla::Shape counts_shape = xla::ShapeUtil::MakeShape(xla::S64, {num_elements});
  unique_elements_shape.set_dynamic_dimension(0, true);
  counts_shape.set_dynamic_dimension(0, true);
  return xla::ShapeUtil::MakeTupleShape(
      {unique_elements_shape, inverse_indices_shape, counts_shape});
}

}  // namespace

Unique2::Unique2(const torch::lazy::Value& input, bool sorted,
                 bool return_inverse, bool return_counts)
    : XlaNode(torch::lazy::OpKind(at::aten::_unique2), {input},
              [&]() {
                return NodeOutputShape(input, return_inverse, return_counts);
              },
              /*num_outputs=*/3,
              torch::lazy::MHash(return_inverse, return_counts)),
      sorted_(sorted),  // sorted is always true for xla backend
      return_inverse_(return_inverse),
      return_counts_(return_counts) {}

torch::lazy::NodePtr Unique2::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Unique2>(operands.at(0), sorted_,
                                        return_inverse_, return_counts_);
}

xla::XlaComputation MakeScatterUpdateComputation(
    xla::PrimitiveType element_type) {
  xla::XlaBuilder cb("ScatterCombiner");
  xla::Shape xla_scalar_shape = xla::ShapeUtil::MakeShape(element_type, {});
  xla::XlaOp p0 = xla::Parameter(&cb, 0, xla_scalar_shape, "p0");
  xla::XlaOp result = xla::Parameter(&cb, 1, xla_scalar_shape, "p1");
  return ConsumeValue(cb.Build(result));
}

xla::XlaComputation MakeScatterCountComputation(
    xla::PrimitiveType element_type) {
  xla::XlaBuilder cb("ScatterCombiner");
  xla::Shape xla_scalar_shape = xla::ShapeUtil::MakeShape(element_type, {});
  xla::XlaOp p0 = xla::Parameter(&cb, 0, xla_scalar_shape, "p0");
  xla::XlaOp result = xla::Parameter(&cb, 1, xla_scalar_shape, "p1");
  result = xla::Add(p0, xla::One(&cb, element_type));
  return ConsumeValue(cb.Build(result));
}

XlaOpVector Unique2::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaBuilder* builder = input.builder();
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::XlaOp input_flattened = XlaHelpers::Flatten(input);
  xla::XlaOp indices =
      xla::Iota(builder, xla::PrimitiveType::S32, num_elements);
  xla::XlaComputation comparator = xla::CreateScalarLtComputation(
      {input_shape.element_type(), xla::PrimitiveType::S32}, builder);
  xla::XlaOp sorted = xla::Sort({input_flattened, indices}, comparator);
  xla::XlaOp sorted_elements = xla::GetTupleElement(sorted, 0);
  xla::XlaOp sorted_indices = xla::GetTupleElement(sorted, 1);
  // adjacent difference
  xla::XlaOp right = xla::Slice(sorted_elements, {1}, {num_elements}, {1});
  xla::XlaOp left = xla::Slice(sorted_elements, {0}, {num_elements - 1}, {1});
  xla::XlaOp diff =
      xla::ConvertElementType(xla::Ne(right, left), xla::PrimitiveType::S32);
  xla::XlaOp adjacent_diff =
      xla::Pad(diff, xla::Zero(builder, xla::PrimitiveType::S32),
               xla::MakeEdgePaddingConfig({{1, 0}}));
  xla::XlaOp cumsum = xla::ReduceWindowWithGeneralPadding(
      adjacent_diff, xla::Zero(builder, xla::PrimitiveType::S32),
      XlaHelpers::CreateAddComputation(xla::PrimitiveType::S32), {num_elements},
      {1},
      /*base_dilations=*/{}, /*window_dilations=*/{}, {{num_elements - 1, 0}});
  xla::ScatterDimensionNumbers scatter_dnums;
  //   operand = s32[3,3] parameter(0)
  //   indices = s32[2] parameter(1)
  //   updates = s32[3,2] parameter(2)
  //   scatter = s32[3,3] scatter(operand, indices, updates),
  //       to_apply=update_computation,
  //       update_window_dims={0},
  //       inserted_window_dims={1},
  //       scatter_dims_to_operand_dims={1},
  //       index_vector_dim=1

  // Example of a 1-D scatter that updates two [1,3] tensors in a tensor of
  // shape [3,3]:
  //
  //   operand = s32[3,3] parameter(0)
  //   indices = s32[2] parameter(1)
  //   updates = s32[2,3] parameter(2)
  //   scatter = s32[3,3] scatter(operand, indices, updates),
  //       to_apply=update_computation,
  //       update_window_dims={1},
  //       inserted_window_dims={0},
  //       scatter_dims_to_operand_dims={0},
  //       index_vector_dim=1
  scatter_dnums.set_index_vector_dim(1);
  scatter_dnums.add_inserted_window_dims(0);
  scatter_dnums.add_scatter_dims_to_operand_dims(0);
  scatter_dnums.add_update_window_dims(1);
  xla::XlaOp sorted_elements_reshaped =
      xla::Reshape(sorted_elements, {num_elements, 1});
  xla::XlaOp unique_elements = xla::Scatter(
      xla::Zeros(builder, xla::ShapeUtil::MakeShape(input_shape.element_type(),
                                                    {num_elements, 1})),
      cumsum, sorted_elements_reshaped,
      MakeScatterUpdateComputation(input_shape.element_type()), scatter_dnums,
      /*indices_are_sorted=*/true, /*unique_indices=*/false);
  xla::XlaOp unique_elements_reshaped = XlaHelpers::Flatten(unique_elements);
  xla::XlaOp cumsum_reshaped = xla::Reshape(cumsum, {num_elements, 1});
  xla::XlaOp inverse_index = xla::Scatter(
      xla::Zeros(builder, xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32,
                                                    {num_elements, 1})),
      sorted_indices, cumsum_reshaped,
      MakeScatterUpdateComputation(xla::PrimitiveType::S32), scatter_dnums,
      /*indices_are_sorted=*/false, /*unique_indices=*/true);
  xla::XlaOp inverse_index_reshaped = xla::Reshape(
      XlaHelpers::Flatten(inverse_index), input_shape.dimensions());

  xla::XlaOp counts = xla::Scatter(
      xla::Zeros(builder, xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32,
                                                    {num_elements, 1})),
      cumsum, cumsum_reshaped,
      MakeScatterCountComputation(xla::PrimitiveType::S32), scatter_dnums,
      /*indices_are_sorted=*/true, /*unique_indices=*/false);

  xla::XlaOp counts_reshaped = XlaHelpers::Flatten(counts);
  xla::XlaOp num_unique_elements =
      xla::Reduce(adjacent_diff, xla::Zero(builder, xla::PrimitiveType::S32),
                  XlaHelpers::CreateAddComputation(xla::PrimitiveType::S32),
                  {0}) +
      xla::One(builder, xla::PrimitiveType::S32);

  std::vector<xla::XlaOp> output = {
      xla::SetDimensionSize(unique_elements_reshaped, num_unique_elements, 0),
      inverse_index_reshaped,
      xla::SetDimensionSize(counts_reshaped, num_unique_elements, 0)};
  return ReturnOps(output, loctx);
}  // namespace torch_xla

std::string Unique2::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", sorted=(" << sorted_ << "), return_inverse=("
     << return_inverse_ << "), return_counts=" << return_counts_;
  return ss.str();
}

}  // namespace torch_xla
