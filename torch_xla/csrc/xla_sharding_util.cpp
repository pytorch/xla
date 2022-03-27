
#include "torch_xla/csrc/xla_sharding_util.h"

#include <unordered_map>

namespace torch_xla {
namespace ir {
namespace {

using xla::internal::XlaBuilderFriend;

}  // namespace

void ShardingUtil::SetHloSharding(const ir::LoweringContext* lowering_ctx) {
  for (std::pair<ir::Output, xla::XlaOp> elem :
       lowering_ctx->GetEmittedOutputs()) {
    const ir::Node* node = elem.first.node;
    auto instruction = XlaBuilderFriend::GetInstruction(elem.second);
    if (node->GetSharding() != nullptr) {
      *instruction->mutable_sharding() = *node->GetSharding();
    }
  }
}

}  // namespace ir
}  // namespace torch_xla
