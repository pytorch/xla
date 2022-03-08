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

std::vector<int64_t> TransposePermutation(int64_t rank) {
  auto dims = std::vector<int64_t>(rank);
  std::iota(dims.begin(), dims.end(), 0);

  // Flip only the last two dimensions
  dims[rank - 1] = rank - 2;
  dims[rank - 2] = rank - 1;

  return dims;
}

std::pair<xla::Shape, xla::Shape> UVShapes(const xla::Shape& input_shape, bool full_matrices, bool compute_uv, bool deprecated_svd) {
  // The input tensor is (..., M, N)
  int64_t m_dim = input_shape.dimensions(input_shape.rank() - 2);
  int64_t n_dim = input_shape.dimensions(input_shape.rank() - 1);

  if (!compute_uv && !deprecated_svd) {
    return std::make_pair(
      xla::ShapeUtil::MakeShape(input_shape.element_type(), {0}),
      xla::ShapeUtil::MakeShape(input_shape.element_type(), {0}));
  }

  xla::Shape u_shape(input_shape);
  if (full_matrices || (!compute_uv && deprecated_svd)) {
    u_shape.set_dimensions(input_shape.rank() - 1, m_dim);
  } else {
    u_shape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
  }

  xla::Shape v_shape(input_shape);
  if (full_matrices) {
    v_shape.set_dimensions(input_shape.rank() - 2, n_dim);
  } else if (!deprecated_svd) {
    v_shape.set_dimensions(input_shape.rank() - 2,
                          full_matrices ? m_dim : std::min(m_dim, n_dim));
  } else {
    v_shape.set_dimensions(input_shape.rank() - 2, n_dim);
    v_shape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
  }

  return std::make_pair(u_shape, v_shape);
}

std::vector<xla::XlaOp> LowerSVD(xla::XlaOp input, bool full_matrices,
                                 bool compute_uv, bool deprecated_svd) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;

  xla::Shape u_shape, v_shape;
  std::tie(u_shape, v_shape) = UVShapes(input_shape, full_matrices, compute_uv, deprecated_svd);

  if (xla::ShapeUtil::IsZeroElementArray(input_shape)) {
    xla::Shape d_shape(input_shape);
    d_shape.DeleteDimension(d_shape.rank() - 1);
    int64_t m_dim = input_shape.dimensions(input_shape.rank() - 2);
    int64_t n_dim = input_shape.dimensions(input_shape.rank() - 1);
    d_shape.set_dimensions(d_shape.rank() - 1, std::min(m_dim, n_dim));

    return {
      xla::Zeros(input.builder(), u_shape),
      xla::Zeros(input.builder(), d_shape),
      xla::Zeros(input.builder(), v_shape)};
  }

  xla::SVDResult svd_result =
      xla::SVD(input, /*max_iter=*/100, /*epsilon=*/1e-6,
               XlaHelpers::mat_mul_precision());
  xla::XlaOp u = svd_result.u;
  xla::XlaOp v = svd_result.v;

  int64_t v_rank = XlaHelpers::ShapeOfXlaOp(v).rank();
  auto perm = TransposePermutation(v_rank);

  if (!compute_uv) {
    u = xla::Zeros(input.builder(), u_shape);
    v = xla::Zeros(input.builder(), v_shape);
  } else if (!full_matrices) {
    std::vector<int64_t> base_indices(u_shape.rank(), 0);

    auto u_sizes = torch::lazy::ToVector<int64_t>(u_shape.dimensions());
    u = BuildSlice(u, base_indices, u_sizes);

    if (!deprecated_svd) {
      // xla::SVD's v is transposed from our expected output shape
      xla::Shape vt_shape = xla::ShapeUtil::PermuteDimensions(perm, v_shape);
      auto v_sizes = torch::lazy::ToVector<int64_t>(vt_shape.dimensions());
      v = BuildSlice(v, base_indices, v_sizes);
      v = xla::Transpose(v, perm);
    } else {
      auto v_sizes = torch::lazy::ToVector<int64_t>(v_shape.dimensions());
      v = BuildSlice(v, base_indices, v_sizes);
    }
  } else if (full_matrices && !deprecated_svd) {
    v = xla::Transpose(v, perm);
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
