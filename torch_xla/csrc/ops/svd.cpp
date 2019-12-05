#include "torch_xla/csrc/ops/svd.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/svd.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
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
    xla::int64 m_dim = input_shape.dimensions(input_shape.rank() - 2);
    xla::int64 n_dim = input_shape.dimensions(input_shape.rank() - 1);
    std::vector<xla::int64> base_indices(input_shape.rank(), 0);

    auto u_sizes = xla::util::ToVector<xla::int64>(input_shape.dimensions());
    u_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    u = BuildSlice(u, base_indices, u_sizes);

    auto v_sizes = xla::util::ToVector<xla::int64>(input_shape.dimensions());
    v_sizes[input_shape.rank() - 2] = n_dim;
    v_sizes[input_shape.rank() - 1] = std::min(m_dim, n_dim);
    v = BuildSlice(v, base_indices, v_sizes);
  }
  return {u, svd_result.d, v};
}

xla::Shape NodeOutputShape(const Value& input, bool some, bool compute_uv) {
  const xla::Shape& input_shape = input.shape();
  XLA_CHECK_GE(input_shape.rank(), 2) << input_shape;
  // The input tensor is ...,M,N
  xla::int64 m_dim = input_shape.dimensions(input_shape.rank() - 2);
  xla::int64 n_dim = input_shape.dimensions(input_shape.rank() - 1);
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

SVD::SVD(const Value& input, bool some, bool compute_uv)
    : Node(ir::OpKind(at::aten::svd), {input},
           [&]() { return NodeOutputShape(input, some, compute_uv); },
           /*num_outputs=*/3, xla::util::MHash(some, compute_uv)),
      some_(some),
      compute_uv_(compute_uv) {}

NodePtr SVD::Clone(OpList operands) const {
  return MakeNode<SVD>(operands.at(0), some_, compute_uv_);
}

XlaOpVector SVD::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(LowerSVD(input, some_, compute_uv_), loctx);
}

std::string SVD::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", some=" << some_
     << ", compute_uv=" << compute_uv_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
