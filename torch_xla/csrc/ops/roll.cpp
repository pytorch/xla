#include "torch_xla/csrc/ops/roll.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

Roll::Roll(const Value& input, std::vector<int64_t> shifts,
           std::vector<int64_t> dims)
    : Node(torch::lazy::OpKind(at::aten::roll), {input}, input.shape(),
           /*num_outputs=*/1, torch::lazy::MHash(shifts, dims)),
      shifts_(std::move(shifts)),
      dims_(std::move(dims)) {}

torch::lazy::NodePtr Roll::Clone(OpList operands) const {
  return ir::MakeNode<Roll>(operands.at(0), shifts_, dims_);
}

XlaOpVector Roll::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);

  int64_t input_dims = input_shape.dimensions_size();
  int64_t num_dims = dims_.size();

  bool need_flatten = num_dims == 0 ? true : false;

  int64_t step = need_flatten ? 1 : num_dims;
  int64_t input_numel = xla::ShapeUtil::ElementsIn(input_shape);

  for (int64_t i = 0; i != step; ++i) {
    input = need_flatten ? xla::Reshape(input, {input_numel}) : input;

    int64_t cur_dim = need_flatten ? 0 : dims_[i];
    if (cur_dim < 0) {
      cur_dim += input_dims;
    }

    int64_t offset = shifts_[i];
    int64_t dim_size =
        need_flatten ? input_numel : input_shape.dimensions(cur_dim);

    // Adjust large offsets into [0, dim_size). This also makes negative
    // offsets positive.
    offset = ((offset % dim_size) + dim_size) % dim_size;

    // Stack two copies of the dimension, then slice from the calculated
    // offset.
    xla::XlaOp concat =
        xla::ConcatInDim(loctx->builder(), {input, input}, cur_dim);
    std::vector<int64_t> start_indices(
        need_flatten ? 1 : input_shape.dimensions_size(), 0);
    start_indices[cur_dim] = dim_size - offset;
    input = BuildSlice(concat, start_indices,
                       need_flatten ? absl::MakeConstSpan({input_numel})
                                    : input_shape.dimensions());
  }

  input = need_flatten ? xla::Reshape(input, input_shape.dimensions()) : input;

  return ReturnOp(input, loctx);
}

std::string Roll::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", shifts=(" << absl::StrJoin(shifts_, ", ") << ")"
     << ", dims=(" << absl::StrJoin(dims_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
