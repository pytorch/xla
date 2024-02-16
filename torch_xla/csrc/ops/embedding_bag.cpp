#include "torch_xla/csrc/ops/embedding_bag.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "tsl/platform/stacktrace.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/slicing.h"
#include "xla/client/lib/loops.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {
// xla::XlaOp MakeOffsetToBag(xla::XlaOp weight, xla::XlaOp indices,
//                            xla::XlaOp offsets) {
//   xla::XlaOp zeroes = xla::ZerosLike(indices);
//   const xla::Shape& offset_shape = ShapeHelper::ShapeOfXlaOp(offsets);
//   xla::XlaOp ones =
//       Broadcast(One(offsets.builder(), offset_shape.element_type()),
//                 offset_shape.dimensions());
//   xla::XlaOp temp = CreateIndexAdd(zeroes, 0, offsets, ones);
//   xla::XlaOp offsets_to_bag = xla::DynamicUpdateSlice(temp, {0}, {0});

//   const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(offsets_to_bag);
//   xla::XlaOp init = XlaHelpers::ScalarValue<float>(
//       0, input_shape.element_type(), offsets_to_bag.builder());
//   xla::XlaComputation reducer =
//       XlaHelpers::CreateAddComputation(input_shape.element_type());
//   return BuildCumulativeComputation(offsets_to_bag, 0, reducer, init);
// }

// xla::XlaOp MakeBagSizes(xla::XlaOp indices, xla::XlaOp offsets) {
//   const xla::Shape& offset_shape = ShapeHelper::ShapeOfXlaOp(offsets);
//   const xla::Shape& indices_shape = ShapeHelper::ShapeOfXlaOp(indices);
//   int64_t offset_size = offset_shape.dimensions(0);
//   int64_t indices_size = indices_shape.dimensions(0);
//   xla::XlaOp one = xla::ConstantR0<int64_t>(offsets.builder(), 1);
//   xla::XlaOp zero = xla::ConstantR0<int64_t>(offsets.builder(), 0);

//   xla::XlaOp slice1 = xla::DynamicSlice(offsets, {one}, /*slice_sizes=*/{offset_size-1});
//   xla::XlaOp slice0 = xla::DynamicSlice(offsets, {zero}, /*slice_sizes=*/{offset_size-1});
//   xla::XlaOp sizes = xla::Sub(slice1, slice0);

//   xla::XlaOp numel = xla::ConstantR0<int64_t>(offsets.builder(), indices_size);
//   xla::XlaOp slice_size = xla::ConstantR0<int64_t>(offsets.builder(), offset_size-1);
//   xla::XlaOp last_offset = xla::DynamicSlice(offsets, {slice_size}, /*slice_sizes=*/{1});
//   xla::XlaOp last_bag_size = xla::Sub({numel}, last_offset);

//   xla::XlaOp bag_sizes =
//       xla::ConcatInDim(offsets.builder(), {sizes, last_bag_size}, 0);
//   return bag_sizes;
// }

// xla::XlaOp MakeMaxIndices(xla::XlaOp bag_sizes, xla::XlaOp weights, xla::XlaOp offsets) {
//   int64_t num_bags = offsets.Shape().size(0);
//   int64_t weight_dim = weights.Shape().size(1);
//   std::vector<int64_t> sizes = {num_bags, weight_dim};
//   xla::XlaOp max_indices = xla::Zeros(
//       weights.builder(), ShapeUtil::MakeShape(offsets.element_type(), sizes));
//   return max_indices;
// }

std::vector<xla::XlaOp> BuildEmbeddingBag(xla::XlaOp weight, xla::XlaOp indices,
                                          xla::XlaOp offsets,
                                          bool include_last_offset) {
  xla::XlaOp output2 = xla::Log(weight);
  xla::XlaOp output3 = xla::Log(weight);
  xla::XlaOp output4 = xla::Log(weight);
  xla::XlaOp embeddings = xla::TorchIndexSelect(weight, indices, 0);
  int64_t n = ShapeHelper::ShapeOfXlaOp(offsets).dimensions(0);
  int64_t weight_dim = ShapeHelper::ShapeOfXlaOp(weight).dimensions(1);
  int64_t num_embeddings = ShapeHelper::ShapeOfXlaOp(indices).dimensions(0);
  std::vector<xla::Shape> shape_elements = {
      xla::ShapeUtil::MakeShape(xla::S64, {}),
      xla::ShapeUtil::MakeShape(xla::S64, {}),
      xla::ShapeUtil::MakeShape(xla::F32, {num_embeddings, weight_dim}),
      xla::ShapeUtil::MakeShape(xla::F32, {1, weight_dim})};
  xla::Shape result_shape = xla::ShapeUtil::MakeTupleShape(shape_elements);

  xla::XlaComputation condition;
  {
    xla::XlaBuilder builder("condition");
    auto prev = xla::Parameter(&builder, 0, result_shape, "prev");
    auto index = xla::GetTupleElement(prev, 0);
    auto final_value = xla::GetTupleElement(prev,1);
    xla::Lt(index, final_value);
    condition = builder.Build().value();
  }

  xla::XlaComputation body;
  {
    xla::XlaBuilder builder("body");
    auto prev = xla::Parameter(&builder, 0, result_shape, "prev");
    auto index = xla::GetTupleElement(prev, 0);
    auto emb = xla::GetTupleElement(prev, 2);
    auto w = xla::GetTupleElement(prev, 3);

    xla::XlaOp slice = xla::DynamicSlice(
        emb, {index, xla::ConstantR0<int64_t>(&builder, 0)}, {1, 3});
    xla::XlaOp result = xla::Add(w, slice);

    xla::Tuple(&builder,
               {
                   xla::Add(index, xla::ConstantR0<int64_t>(&builder, 1)),
                   xla::GetTupleElement(prev, 1),
                   xla::GetTupleElement(prev, 2),
                   result,
               });
    body = builder.Build().value();
  }

  xla::Array<float> initial_vector({1, weight_dim}, 0.f);
  std::vector<xla::XlaOp> results;
  for (int64_t i = 0; i < n; i++) {
    xla::XlaOp start = xla::DynamicSlice(
        offsets, {xla::ConstantR0<int64_t>(offsets.builder(), i)}, {1});
    if (i == n - 1 && !include_last_offset) continue;
    xla::XlaOp end =
        i == n - 1
            ? xla::ConstantR1<int64_t>(offsets.builder(), 1, num_embeddings)
            : xla::DynamicSlice(
                  offsets, {xla::ConstantR0<int64_t>(offsets.builder(), i + 1)},
                  {1});
    // Create a While node with computations for the condition and the body.
    auto init_tuple = xla::Tuple(
        offsets.builder(),
        {xla::Reshape(start, {0}, {}), xla::Reshape(end, {0}, {}), embeddings,
         xla::ConstantFromArray<float>(offsets.builder(), initial_vector)});
    auto result = xla::While(condition, body, init_tuple);
    results.push_back(xla::GetTupleElement(result, 3));
  };
  xla::XlaOp final = xla::ConcatInDim(offsets.builder(), results, 0);
  return {final, output2, output3, output4};
}

xla::Shape NodeOutputShapes(const torch::lazy::Value& weight,
                            const torch::lazy::Value& indices,
                            const torch::lazy::Value& offsets,
                            bool include_last_offset) {
  auto lower_for_shapes_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      BuildEmbeddingBag(operands[0], operands[1], operands[2],
                                        include_last_offset));
  };

  std::vector<xla::Shape> input_shapes = {
      GetXlaShape(weight), GetXlaShape(indices), GetXlaShape(offsets)};

  return InferOutputShape(absl::MakeSpan(input_shapes), lower_for_shapes_fn);
}
}  // namespace

std::string EmbeddingBag::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  return ss.str();
}

EmbeddingBag::EmbeddingBag(const torch::lazy::Value& weight,
                           const torch::lazy::Value& indices,
                           const torch::lazy::Value& offsets,
                           bool scale_grad_by_freq, int64_t mode, bool sparse,
                           const c10::optional<at::Tensor>& per_sample_weights,
                           bool include_last_offset, int64_t padding_idx)
    : XlaNode(
          torch::lazy::OpKind(at::aten::embedding_bag),
          {weight, indices, offsets},
          [&]() {
            return NodeOutputShapes(weight, indices, offsets,
                                    include_last_offset);
          },
          /*num_outputs=*/4,
          torch::lazy::MHash(scale_grad_by_freq, mode, sparse,
                             include_last_offset, padding_idx)),
      scale_grad_by_freq_(scale_grad_by_freq),
      mode_(mode),
      sparse_(sparse),
      include_last_offset_(include_last_offset),
      padding_idx_(padding_idx) {}

torch::lazy::NodePtr EmbeddingBag::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<EmbeddingBag>(operands.at(0), operands.at(0),
                                             operands.at(0), false, 1, false,
                                             c10::nullopt, false, 0);
}

XlaOpVector EmbeddingBag::Lower(LoweringContext* loctx) const {
  xla::XlaOp weight = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp offsets = loctx->GetOutputOp(operand(2));
  std::vector<xla::XlaOp> ops =
      BuildEmbeddingBag(weight, indices, offsets, include_last_offset_);
  return ReturnOps(absl::MakeSpan(ops), loctx);
}

}  // namespace torch_xla