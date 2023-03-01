#include "torch_xla/csrc/ops/linalg_vector_norm.h"

#include "absl/strings/str_join.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerLinalgVectorNorm(xla::XlaOp input, xla::XlaOp ord,
                                 absl::Span<const int64_t> dimensions,
                                 bool keep_dim,
                                 c10::optional<at::ScalarType> dtype) {
    return BuildLinalgVectorNorm(CastToScalarType(input, dtype), 
                                  dimensions, keep_dim);
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const torch::lazy::Value& ord,
                           absl::Span<const int64_t> dimensions,
                           bool keep_dim,
                           c10::optional<at::ScalarType> dtype) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerLinalgVectorNorm(operands[0], operands[1],
                                 dimensions, keep_dim, dtype);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

LinalgVectorNorm::LinalgVectorNorm(const torch::lazy::Value& input, 
                                   const torch::lazy::Value& ord,
                                   std::vector<int64_t> dimensions,
                                   bool keep_dim, 
                                   c10::optional<at::ScalarType> dtype)
    : XlaNode(torch::lazy::OpKind(at::aten::linalg_vector_norm), 
              {input, ord},
              [&]() {
                return NodeOutputShape(input, ord, dimensions,
                                       keep_dim, dtype);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(dimensions, keep_dim,
                                 torch::lazy::OptionalOr<int>(dtype, -1))),
      dimensions_(std::move(dimensions)),
      keep_dim_(keep_dim),
      dtype_(dtype) {}

torch::lazy::NodePtr LinalgVectorNorm::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<LinalgVectorNorm>(operands.at(0), operands.at(1),
                                                 dimensions_, keep_dim_, dtype_);
}

XlaOpVector LinalgVectorNorm::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp ord = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      LowerLinalgVectorNorm(input, ord, dimensions_, keep_dim_, dtype_), loctx);
}

std::string LinalgVectorNorm::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dimensions=("
     << absl::StrJoin(dimensions_, ", ")
     << "), keep_dim=" << keep_dim_
     << ", dtype=" << torch::lazy::OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace torch_xla
