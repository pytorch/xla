#include "torch_xla/csrc/ops/topk.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input, int64_t k,
                           int64_t dim, bool largest, bool sorted,
                           bool stable) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(),
                      CreateTopK(operands[0], k, dim, largest, stable));
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

xla::Shape NodeOutputShapeSymInt(const torch::lazy::Value& input,
                                 int64_t k_upper_bound, int64_t dim,
                                 bool largest, bool sorted, bool stable) {
  xla::Shape input_shape = GetXlaShape(input);
  std::vector<int64_t> dimensions(input_shape.dimensions().begin(),
                                  input_shape.dimensions().end());
  XLA_CHECK_LT(dim, input_shape.rank());
  dimensions[dim] = k_upper_bound;
  xla::Shape values_shape =
      xla::ShapeUtil::MakeShape(input_shape.element_type(), dimensions);
  xla::Shape indices_shape =
      xla::ShapeUtil::MakeShape(xla::PrimitiveType::S64, dimensions);
  values_shape.set_dynamic_dimension(dim, true);
  indices_shape.set_dynamic_dimension(dim, true);
  return xla::ShapeUtil::MakeTupleShape({values_shape, indices_shape});
}

}  // namespace

TopK::TopK(const torch::lazy::Value& input, int64_t k, int64_t dim,
           bool largest, bool sorted, bool stable)
    : XlaNode(torch::lazy::OpKind(at::aten::topk), {input},
              [&]() {
                return NodeOutputShape(input, k, dim, largest, sorted, stable);
              },
              /*num_outputs=*/2,
              torch::lazy::MHash(k, dim, largest, sorted, stable)),
      k_(k),
      dim_(dim),
      largest_(largest),
      sorted_(sorted),
      stable_(stable) {}

torch::lazy::NodePtr TopK::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<TopK>(operands.at(0), k_, dim_, largest_,
                                     sorted_, stable_);
}

XlaOpVector TopK::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(CreateTopK(input, k_, dim_, largest_, stable_), loctx);
}

std::string TopK::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", k=" << k_ << ", dim=" << dim_
     << ", largest=" << largest_ << ", sorted=" << sorted_
     << ", stable=" << stable_;
  return ss.str();
}

TopKSymInt::TopKSymInt(const torch::lazy::Value& input, const SymIntElements& k,
                       int64_t dim, bool largest, bool sorted, bool stable)
    : XlaNode(torch::lazy::OpKind(at::aten::topk),
              {input, torch::lazy::Value(k.GetSizeNodes().front())},
              [&]() {
                return NodeOutputShapeSymInt(input, k.GetUpperBounds().front(),
                                             dim, largest, sorted, stable);
              },
              /*num_outputs=*/2,
              torch::lazy::MHash(k.GetUpperBounds().front(), dim, largest,
                                 sorted, stable)),
      k_upper_bound_(k.GetUpperBounds().front()),
      dim_(dim),
      largest_(largest),
      sorted_(sorted),
      stable_(stable) {}

XlaOpVector TopKSymInt::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp size_op = loctx->GetOutputOp(operand(1));
  std::vector<xla::XlaOp> results =
      CreateTopK(input, k_upper_bound_, dim_, largest_, stable_);
  std::vector<xla::XlaOp> resized_results;
  std::transform(
      results.begin(), results.end(), std::back_inserter(resized_results),
      [&](xla::XlaOp op) { return xla::SetDimensionSize(op, size_op, dim_); });
  return ReturnOps(resized_results, loctx);
}

std::string TopKSymInt::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", k<=" << k_upper_bound_ << ", dim=" << dim_
     << ", largest=" << largest_ << ", sorted=" << sorted_
     << ", stable=" << stable_;
  return ss.str();
}

}  // namespace torch_xla
