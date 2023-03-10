#include "torch_xla/csrc/ops/svd.h"

#include <torch/csrc/lazy/core/util.h>

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "third_party/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace {

std::vector<xla::XlaOp> LowerSVD(xla::XlaOp input, bool some, bool compute_uv) {
  xla::SVDResult svd_result =
      xla::SVD(input, /*max_iter=*/100, /*epsilon=*/1e-6,
               XlaHelpers::mat_mul_precision());
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp u = svd_result.u;
  xla::XlaOp v = svd_result.v;
  if (!compute_uv) {
    u = xla::Zeros(input.builder(), XlaHelpers::ShapeOfXlaOp(u));
    v = xla::Zeros(input.builder(), XlaHelpers::ShapeOfXlaOp(v));
  } else if (some) {
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
  return {u, svd_result.d, v};
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input, bool some,
                           bool compute_uv) {
  const xla::Shape& input_shape = GetXlaShape(input);
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,M,N
  int64_t m_dim = input_shape.dimensions(input_shape.rank() - 2);
  int64_t n_dim = input_shape.dimensions(input_shape.rank() - 1);
  xla::Shape ushape(input_shape);
  if (!compute_uv || !some) {
    ushape.set_dimensions(input_shape.rank() - 1, m_dim);
  } else {
    ushape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
  }
  // D is min(M, N).
  xla::Shape dshape = xla::ShapeUtil::MakeShape(input_shape.element_type(),
                                                {std::min(m_dim, n_dim)});
  // V is NxN
  xla::Shape vshape(input_shape);
  vshape.set_dimensions(input_shape.rank() - 2, n_dim);
  if (some) {
    vshape.set_dimensions(input_shape.rank() - 1, std::min(m_dim, n_dim));
  }
  return xla::ShapeUtil::MakeTupleShape({ushape, dshape, vshape});
}

}  // namespace

SVD::SVD(const torch::lazy::Value& input, bool some, bool compute_uv)
    : XlaNode(torch::lazy::OpKind(at::aten::svd), {input},
              [&]() { return NodeOutputShape(input, some, compute_uv); },
              /*num_outputs=*/3, torch::lazy::MHash(some, compute_uv)),
      some_(some),
      compute_uv_(compute_uv) {}

torch::lazy::NodePtr SVD::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<SVD>(operands.at(0), some_, compute_uv_);
}

XlaOpVector SVD::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerSVD(input, some_, compute_uv_), loctx);
}

std::string SVD::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", some=" << some_
     << ", compute_uv=" << compute_uv_;
  return ss.str();
}

}  // namespace torch_xla
