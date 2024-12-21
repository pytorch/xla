#include "torch_xla/csrc/ops/eigh.h"

#include <array>

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"
#include "xla/client/lib/self_adjoint_eig.h"

namespace torch_xla {

namespace {

std::array<xla::XlaOp, 2> LowerImpl(xla::XlaOp input, bool lower) {
  auto [eigenvectors, eigenvalues] =
      // The default `max_iter` and `tol` values lead to very inaccurate
      // decomposition. To improve accuracy we run more iterations and tighter
      // tolerance. These values are taken from the JAX lowering of eigh:
      // https://github.com/google/jax/blob/a8b425cac50c842f66f36903dfb93fe6ad5a2a5b/jax/_src/lax/linalg.py#L726
      xla::SelfAdjointEig(input, lower, /* max_iter */ 100, /* tol */ 1e-6);
  // Torch expects `(eigenvalues, eigenvectors)` and XLA returns the reverse.
  return {eigenvalues, eigenvectors};
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::Tuple(operands[0].builder(), LowerImpl(operands[0], true));
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Eigh::Eigh(const torch::lazy::Value& input, std::string_view uplo)
    : XlaNode(
          torch::lazy::OpKind(at::aten::_linalg_eigh), {input},
          [&]() { return NodeOutputShape(input); },
          /*num_outputs=*/2, torch::lazy::MHash(uplo)) {
  XLA_CHECK(uplo == "L" || uplo == "U") << "Expected L or U, got: " << uplo;
  uplo_ = uplo[0];
}

XlaOpVector Eigh::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  bool lower = uplo_ == 'L';
  return ReturnOps(LowerImpl(input, lower), loctx);
}

std::string Eigh::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", uplo=" << uplo_;
  return ss.str();
}

}  // namespace torch_xla
