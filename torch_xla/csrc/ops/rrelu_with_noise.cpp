#include "torch_xla/csrc/ops/rrelu_with_noise.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

RreluWithNoise::RreluWithNoise(const Value& input, at::Scalar lower,
                               at::Scalar upper, bool training)
    : Node(ir::OpKind(at::aten::rrelu_with_noise), {input},
           xla::ShapeUtil::MakeTupleShape({input.shape(), input.shape()}),
           /*num_outputs=*/2,
           xla::util::MHash(ScalarHash(lower), ScalarHash(upper), training)),
      lower_(std::move(lower)),
      upper_(std::move(upper)),
      training_(training) {}

NodePtr RreluWithNoise::Clone(OpList operands) const {
  return MakeNode<RreluWithNoise>(operands.at(0), lower_, upper_, training_);
}

XlaOpVector RreluWithNoise::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOps(BuildRrelu(input, lower_, upper_, training_), loctx);
}

std::string RreluWithNoise::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lower=" << lower_ << ", upper=" << upper_
     << ", training=" << training_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
