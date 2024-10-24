#include "torch_xla/csrc/ops/embedding_bag.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/shape_helper.h"
#include "torch_xla/csrc/xla_lower_util.h"
#include "tsl/platform/stacktrace.h"
#include "xla/client/lib/constants.h"
#include "xla/client/lib/slicing.h"
#include "xla/hlo/builder/lib/loops.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {
const int MODE_SUM = 0;
const int MODE_MEAN = 1;
const int MODE_MAX = 2;
std::vector<xla::XlaOp> BuildEmbeddingBag(xla::XlaOp weight, xla::XlaOp indices,
                                          xla::XlaOp offsets,
                                          xla::XlaOp per_sample_weights,
                                          bool include_last_offset, int mode) {
  xla::Shape offset_shape = ShapeHelper::ShapeOfXlaOp(offsets);
  int64_t n = offset_shape.dimensions(0);
  xla::Shape weight_shape = ShapeHelper::ShapeOfXlaOp(weight);
  int64_t weight_dim = weight_shape.dimensions(1);
  xla::Shape indices_shape = ShapeHelper::ShapeOfXlaOp(indices);
  int64_t num_embeddings = indices_shape.dimensions(0);
  XLA_CHECK(indices_shape.rank() == 1 || indices_shape.rank() == 2)
      << "input has to be a 1D or 2D Tensor, but got Tensor of dimension "
      << indices_shape.rank();
  if (indices_shape.rank() == 1) {
    XLA_CHECK(offset_shape.rank() == 1)
        << "offsets has to be a 1D Tensor, but got Tensor of dimension "
        << offset_shape.rank();
  }
  XLA_CHECK(weight_shape.rank() == 2)
      << "weight has to be a 2D Tensor, but got Tensor of dimension "
      << weight_shape.rank();

  xla::XlaOp output2 = xla::ZerosLike(indices);
  xla::XlaOp output3 = xla::ZerosLike(offsets);
  std::vector<int64_t> sizes = {n, weight_dim};
  xla::XlaOp output4 =
      xla::Zeros(offsets.builder(),
                 xla::ShapeUtil::MakeShape(offset_shape.element_type(), sizes));

  xla::XlaOp embeddings = xla::TorchIndexSelect(weight, indices, 0);
  xla::XlaOp embeddings_weighted = xla::Mul(
      embeddings, xla::ConvertElementType(
                      xla::BroadcastInDim(per_sample_weights,
                                          {num_embeddings, weight_dim}, {0}),
                      weight_shape.element_type()));

  std::vector<xla::Shape> shape_elements = {
      xla::ShapeUtil::MakeShape(offset_shape.element_type(), {}),
      xla::ShapeUtil::MakeShape(offset_shape.element_type(), {}),
      xla::ShapeUtil::MakeShape(weight_shape.element_type(),
                                {num_embeddings, weight_dim}),
      xla::ShapeUtil::MakeShape(weight_shape.element_type(), {1, weight_dim})};
  xla::Shape result_shape = xla::ShapeUtil::MakeTupleShape(shape_elements);

  xla::XlaComputation condition;
  {
    xla::XlaBuilder builder("condition");
    auto prev = xla::Parameter(&builder, 0, result_shape, "prev");
    auto index = xla::GetTupleElement(prev, 0);
    auto final_value = xla::GetTupleElement(prev, 1);
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
        emb,
        {index, xla::ConvertElementType(xla::ConstantR0<int64_t>(&builder, 0),
                                        offset_shape.element_type())},
        {1, weight_dim});
    xla::XlaOp result =
        mode == MODE_SUM ? xla::Add(w, slice) : xla::Max(w, slice);

    xla::Tuple(&builder,
               {
                   xla::Add(index, xla::ConvertElementType(
                                       xla::ConstantR0<int64_t>(&builder, 1),
                                       offset_shape.element_type())),
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
    if (i == n - 1 && include_last_offset) continue;
    xla::XlaOp end =
        i == n - 1 && !include_last_offset
            ? xla::ConvertElementType(xla::ConstantR1<int64_t>(
                                          offsets.builder(), 1, num_embeddings),
                                      offset_shape.element_type())
            : xla::DynamicSlice(
                  offsets, {xla::ConstantR0<int64_t>(offsets.builder(), i + 1)},
                  {1});
    // Create a While node with computations for the condition and the body.
    auto init_tuple = xla::Tuple(
        offsets.builder(),
        {xla::Reshape(start, {0}, {}), xla::Reshape(end, {0}, {}),
         embeddings_weighted,
         xla::ConvertElementType(
             xla::ConstantFromArray<float>(offsets.builder(), initial_vector),
             weight_shape.element_type())});
    auto result = xla::While(condition, body, init_tuple);
    results.push_back(xla::GetTupleElement(result, 3));
  };
  xla::XlaOp output1 = xla::ConcatInDim(offsets.builder(), results, 0);
  return {output1, output2, output3, output4};
}

xla::Shape NodeOutputShapes(const torch::lazy::Value& weight,
                            const torch::lazy::Value& indices,
                            const torch::lazy::Value& offsets,
                            const torch::lazy::Value& per_sample_weights,
                            bool include_last_offset, bool mode) {
  auto lower_for_shapes_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(
        operands[0].builder(),
        BuildEmbeddingBag(operands[0], operands[1], operands[2], operands[3],
                          include_last_offset, mode));
  };

  std::vector<xla::Shape> input_shapes = {
      GetXlaShape(weight), GetXlaShape(indices), GetXlaShape(offsets),
      GetXlaShape(per_sample_weights)};

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
                           const torch::lazy::Value& offsets, int64_t mode,
                           const torch::lazy::Value& per_sample_weights,
                           bool include_last_offset)
    : XlaNode(
          torch::lazy::OpKind(at::aten::embedding_bag),
          {weight, indices, offsets, per_sample_weights},
          [&]() {
            return NodeOutputShapes(weight, indices, offsets,
                                    per_sample_weights, include_last_offset,
                                    mode);
          },
          /*num_outputs=*/4, torch::lazy::MHash(mode, include_last_offset)),
      mode_(mode),
      include_last_offset_(include_last_offset) {}

torch::lazy::NodePtr EmbeddingBag::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<EmbeddingBag>(operands.at(0), operands.at(1),
                                           operands.at(2), mode_,
                                           operands.at(3), false);
}

XlaOpVector EmbeddingBag::Lower(LoweringContext* loctx) const {
  xla::XlaOp weight = loctx->GetOutputOp(operand(0));
  xla::XlaOp indices = loctx->GetOutputOp(operand(1));
  xla::XlaOp offsets = loctx->GetOutputOp(operand(2));
  xla::XlaOp per_sample_weights = loctx->GetOutputOp(operand(3));
  std::vector<xla::XlaOp> ops =
      BuildEmbeddingBag(weight, indices, offsets, per_sample_weights,
                        include_last_offset_, mode_);
  return ReturnOps(absl::MakeSpan(ops), loctx);
}

}  // namespace torch_xla