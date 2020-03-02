#include "torch_xla/csrc/ops/device_data.h"

#include <sstream>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

DeviceData::DeviceData(std::shared_ptr<xla::ComputationClient::Data> data)
    : Node(xla_device_data, data->shape(), /*num_outputs=*/1,
           /*hash_seed=*/101),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", device=" << data_->device();
  return ss.str();
}

NodePtr DeviceData::Clone(OpList operands) const {
  return MakeNode<DeviceData>(data_);
}

XlaOpVector DeviceData::Lower(LoweringContext* loctx) const {
  return ReturnOp(loctx->GetParameter(data_), loctx);
}

DeviceData* DeviceData::Cast(const Node* node) {
  return NodeCast<DeviceData>(node, xla_device_data);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
