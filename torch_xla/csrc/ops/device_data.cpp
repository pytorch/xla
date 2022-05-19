#include "torch_xla/csrc/ops/device_data.h"

#include <sstream>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {

DeviceData::DeviceData(std::shared_ptr<torch::lazy::BackendData> data)
    : XlaNode(xla_device_data, UnwrapXlaData(data)->shape(), /*num_outputs=*/1,
              /*hash_seed=*/(uint32_t)101),
      data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", device=" << data_->device();
  return ss.str();
}

torch::lazy::NodePtr DeviceData::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<DeviceData>(data_);
}

XlaOpVector DeviceData::Lower(LoweringContext* loctx) const {
  return ReturnOp(loctx->GetParameter(data_), loctx);
}

DeviceData* DeviceData::Cast(const torch::lazy::Node* node) {
  return torch_xla::NodeCast<DeviceData>(node, xla_device_data);
}

}  // namespace torch_xla
