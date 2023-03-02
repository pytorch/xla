#include "torch_xla/csrc/ops/cdist.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& x1,
                           const torch::lazy::Value& x2,
                           const torch::lazy::Value& p, bool use_hamming,
                           bool use_chebyshev) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildCdistForward(operands[0], operands[1], operands[2], use_hamming,
                             use_chebyshev);
  };
  return InferOutputShape({GetXlaShape(x1), GetXlaShape(x2), GetXlaShape(p)},
                          lower_for_shape_fn);
}

}  // namespace

CdistForward::CdistForward(const torch::lazy::Value& x1,
                           const torch::lazy::Value& x2,
                           const torch::lazy::Value& p, bool use_hamming,
                           bool use_chebyshev)
    : XlaNode(
          torch::lazy::OpKind(at::aten::_cdist_forward), {x1, x2, p},
          [&]() {
            return NodeOutputShape(x1, x2, p, use_hamming, use_chebyshev);
          },
          /*num_outputs=*/1, torch::lazy::MHash(use_hamming, use_chebyshev)),
      use_hamming_(use_hamming),
      use_chebyshev_(use_chebyshev) {}

torch::lazy::NodePtr CdistForward::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CdistForward>(operands.at(0), operands.at(1),
                                             operands.at(2), use_hamming_,
                                             use_chebyshev_);
}

XlaOpVector CdistForward::Lower(LoweringContext* loctx) const {
  xla::XlaOp x1 = loctx->GetOutputOp(operand(0));
  xla::XlaOp x2 = loctx->GetOutputOp(operand(1));
  xla::XlaOp p = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildCdistForward(x1, x2, p, use_hamming_, use_chebyshev_),
                  loctx);
}

std::string CdistForward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", use_hamming=" << use_hamming_
     << ", use_chebyshev=" << use_chebyshev_;
  return ss.str();
}

}  // namespace torch_xla
