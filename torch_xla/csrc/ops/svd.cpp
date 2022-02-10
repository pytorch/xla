#include "torch_xla/csrc/ops/svd.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

std::vector<xla::XlaOp> LowerSVD(xla::XlaOp input, bool full_matrices,
                                 bool compute_uv, bool deprecated_svd) {
  xla::SVDResult svd_result =
      xla::SVD(input, /*max_iter=*/100, /*epsilon=*/1e-6,
               XlaHelpers::mat_mul_precision());
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp u = svd_result.u;
  xla::XlaOp v = svd_result.v;
  if (!compute_uv) {
    xla::Shape u_shape = deprecated_svd ? XlaHelpers::ShapeOfXlaOp(u)
                                        : xla::ShapeUtil::MakeShape(
                                              input_shape.element_type(), {0});
    xla::Shape v_shape = deprecated_svd ? XlaHelpers::ShapeOfXlaOp(v)
                                        : xla::ShapeUtil::MakeShape(
                                              input_shape.element_type(), {0});
    u = xla::Zeros(input.builder(), u_shape);
    v = xla::Zeros(input.builder(), v_shape);
  } else if (!full_matrices) {
    int64_t m_dim = input_shape.dimensions(input_shape.rank() - 2);
    int64_t n_dim = input_shape.dimensions(input_shape.rank() - 1);
    std::vector<int64_t> base_indices(input_shape.rank(), 0);

    auto u_sizes = torch::lazy::ToVector<int64_t>(input_shape.dimensions());
    u_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    u = BuildSlice(u, base_indices, u_sizes);

    auto v_sizes = torch::lazy::ToVector<int64_t>(input_shape.dimensions());
    v_sizes[input_shape.rank() - 2] = n_dim;
    v_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    v = BuildSlice(v, base_indices, v_sizes);
  }

  // Return Vh (conjugate transpose) for torch.linalg.svd. Conjugate not lowered
  // yet so just transpose.
  if (compute_uv && !deprecated_svd) {
    int64_t v_rank = XlaHelpers::ShapeOfXlaOp(v).rank();
    auto dims = std::vector<int64_t>(v_rank);
    std::iota(dims.begin(), dims.end(), 0);

    dims[v_rank - 2] = v_rank - 1;
    dims[v_rank - 1] = v_rank - 2;
    v = xla::Transpose(v, dims);
  }

  return {u, svd_result.d, v};
}

xla::Shape NodeOutputShape(const Value& input, bool full_matrices,
                           bool compute_uv, bool deprecated_svd) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    std::vector<xla::XlaOp> values = LowerSVD(operands[0], full_matrices, compute_uv, deprecated_svd);
    return xla::Tuple(operands[0].builder(), values);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

SVD::SVD(const Value& input, bool full_matrices, bool compute_uv,
         bool deprecated_svd)
    : Node(torch::lazy::OpKind(at::aten::svd), {input},
           [&]() {
             return NodeOutputShape(input, full_matrices, compute_uv,
                                    deprecated_svd);
           },
           /*num_outputs=*/3,
           torch::lazy::MHash(full_matrices, compute_uv, deprecated_svd)),
      full_matrices_(full_matrices),
      compute_uv_(compute_uv),
      deprecated_svd_(deprecated_svd) {}

NodePtr SVD::Clone(OpList operands) const {
  return MakeNode<SVD>(operands.at(0), full_matrices_, compute_uv_,
                       deprecated_svd_);
}

XlaOpVector SVD::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(
      LowerSVD(input, full_matrices_, compute_uv_, deprecated_svd_), loctx);
}

std::string SVD::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", full_matrices=" << full_matrices_
     << ", compute_uv=" << compute_uv_
     << ", deprecated_svd=" << deprecated_svd_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
