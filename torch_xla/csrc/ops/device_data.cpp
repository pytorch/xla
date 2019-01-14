#include "ops/device_data.h"

#include <sstream>

#include "lowering_context.h"
#include "ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

DeviceData::DeviceData(std::shared_ptr<xla::ComputationClient::Data> data)
    : Node(xla_device_data, {}, data->shape()), data_(std::move(data)) {}

std::string DeviceData::ToString() const {
  std::stringstream ss;
  ss << "DeviceData(" << data_->device() << ", " << data_->shape() << ")";
  return ss.str();
}

XlaOpVector DeviceData::Lower(LoweringContext* loctx) const {
  return ReturnOp(loctx->GetParameter(data_), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
