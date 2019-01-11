#pragma once

#include "ir.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"

namespace torch_xla {
namespace ir {
namespace ops {

class DeviceData : public Node {
 public:
  DeviceData(std::shared_ptr<xla::ComputationClient::Data> data);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  std::shared_ptr<xla::ComputationClient::Data> data_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
