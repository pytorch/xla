#include "torch_xla/csrc/ops/custom_sharding.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {
std::string TypeToString(const CustomSharding::Type& type) {
  switch (type) {
    case CustomSharding::Type::kSharding:
      return "Sharding";
    case CustomSharding::Type::kSPMDFullToShardShape:
      return "SPMDFullToShardShape";
    case CustomSharding::Type::kSPMDShardToFullShape:
      return "SPMDShardToFullShape";
  }
}
}  // namespace

CustomSharding::CustomSharding(const torch::lazy::Value& input,
                               const xla::Shape& output_shape,
                               const CustomSharding::Type& type)
    : XlaNode(xla_custom_sharding, {input}, output_shape,
              /*num_outputs=*/1, torch::lazy::MHash(static_cast<int>(type))),
      type(type),
      output_shape(output_shape) {}

torch::lazy::NodePtr CustomSharding::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CustomSharding>(operands.at(0), output_shape,
                                               type);
}

XlaOpVector CustomSharding::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      BuildCustomSharding(input, TypeToString(type), output_shape);
  return ReturnOp(output, loctx);
}

std::string CustomSharding::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", " << TypeToString(type);
  return ss.str();
}

}  // namespace torch_xla
