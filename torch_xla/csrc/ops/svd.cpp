#include "torch_xla/csrc/ops/svd.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

std::vector<xla::XlaOp> LowerSVD(xla::XlaOp input, bool full_matrices, bool compute_uv, bool deprecated_svd) {
  xla::SVDResult svd_result =
      xla::SVD(input, /*max_iter=*/100, /*epsilon=*/1e-6,
               XlaHelpers::mat_mul_precision());
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp u = svd_result.u;
  xla::XlaOp v = svd_result.v;
  if (!compute_uv) {
    xla::Shape u_shape = deprecated_svd ? XlaHelpers::ShapeOfXlaOp(u) : xla::ShapeUtil::MakeShape(input_shape.element_type(), {0});
    xla::Shape v_shape = deprecated_svd ? XlaHelpers::ShapeOfXlaOp(v) :  xla::ShapeUtil::MakeShape(input_shape.element_type(), {0});
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

xla::Shape NodeOutputShape(const Value& input, bool full_matrices, bool compute_uv, bool deprecated_svd) {
  const xla::Shape& input_shape = input.shape();
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,M,N
  int64_t m_dim = input_shape.dimensions(input_shape.rank() - 2);
  int64_t n_dim = input_shape.dimensions(input_shape.rank() - 1);
  xla::Shape ushape(input_shape);

  if (!compute_uv && !deprecated_svd) {
    ushape = xla::ShapeUtil::MakeShape(input_shape.element_type(), {0});
  } else if (!compute_uv || full_matrices) {
    ushape.set_dimensions(input_shape.rank() - 1, m_dim);
  } else {
    ushape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
  }
  // D is min(M, N).
  xla::Shape dshape = xla::ShapeUtil::MakeShape(input_shape.element_type(),
                                                {std::min(m_dim, n_dim)});
  // V is NxN
  xla::Shape vshape(input_shape);
  if (!deprecated_svd) {
    if (compute_uv) {
      vshape.set_dimensions(input_shape.rank() - 1, n_dim);
      vshape.set_dimensions(input_shape.rank() - 2,
                            full_matrices ? m_dim : std::min(m_dim, n_dim));
    } else {
      vshape = xla::ShapeUtil::MakeShape(input_shape.element_type(), {0});
    }
  } else {
    vshape.set_dimensions(input_shape.rank() - 2, n_dim);
    if (!full_matrices) {
      vshape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
    }
  }

  return xla::ShapeUtil::MakeTupleShape({ushape, dshape, vshape});
}

}  // namespace

SVD::SVD(const Value& input, bool full_matrices, bool compute_uv, bool deprecated_svd)
    : Node(torch::lazy::OpKind(at::aten::svd), {input},
           [&]() { return NodeOutputShape(input, full_matrices, compute_uv, deprecated_svd); },
           /*num_outputs=*/3, torch::lazy::MHash(full_matrices, compute_uv, deprecated_svd)),
      full_matrices_(full_matrices),
      compute_uv_(compute_uv),
      deprecated_svd_(deprecated_svd) {}

NodePtr SVD::Clone(OpList operands) const {
  return MakeNode<SVD>(operands.at(0), full_matrices_, compute_uv_, deprecated_svd_);
}

XlaOpVector SVD::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerSVD(input, full_matrices_, compute_uv_, deprecated_svd_), loctx);
}

std::string SVD::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", full_matrices=" << full_matrices_
     << ", compute_uv=" << compute_uv_ << ", deprecated_svd=" << deprecated_svd_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
