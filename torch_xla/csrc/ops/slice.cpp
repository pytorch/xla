#include "torch_xla/csrc/ops/slice.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerSlice(const xla::XlaOp& input, xla::int64 dim, xla::int64 start,
                      xla::int64 end, xla::int64 step) {
  xla::Shape shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<xla::int64> start_indices(shape.rank());
  std::vector<xla::int64> limit_indices(shape.dimensions());
  std::vector<xla::int64> strides(shape.rank(), 1);
  start_indices.at(dim) = start;
  limit_indices.at(dim) = end;
  strides.at(dim) = step;
  return xla::Slice(input, start_indices, limit_indices, strides);
}

xla::Shape NodeOutputShape(const Value& input, xla::int64 dim, xla::int64 start,
                           xla::int64 end, xla::int64 step) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp { return LowerSlice(operands[0], dim, start, end, step); };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Slice::Slice(const Value& input, xla::int64 dim, xla::int64 start,
             xla::int64 end, xla::int64 step)
    : Node(
          ir::OpKind(at::aten::slice), {input},
          [&]() { return NodeOutputShape(input, dim, start, end, step); },
          /*num_outputs=*/1, xla::util::MHash(dim, start, end, step)),
      dim_(dim),
      start_(start),
      end_(end),
      step_(step) {}

XlaOpVector Slice::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerSlice(input, dim_, start_, end_, step_), loctx);
}

std::string Slice::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", step=" << step_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
