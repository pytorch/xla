#ifndef XLA_TORCH_XLA_CSRC_XLA_DATA_H_
#define XLA_TORCH_XLA_CSRC_XLA_DATA_H_

#include <torch/csrc/lazy/backend/backend_data.h>
#include <torch/csrc/lazy/backend/backend_device.h>
#include <torch/csrc/lazy/core/shape.h>

#include "third_party/xla_client/computation_client.h"

namespace torch_xla {

// To be noted, the most appropriate way to adopt BackendData
// should actually be letting ComputationClient::Data inherit it.
// Since ComputationClient is within TensorFlow, and TF cannot
// depend on PyTorch. Therefore, we have this intermediate wrapper.
class XLAData : public torch::lazy::BackendData {
 public:
  XLAData(const torch::lazy::Shape& shape,
          const torch::lazy::BackendDevice& device,
          xla::ComputationClient::DataPtr xla_data);

  XLAData(xla::ComputationClient::DataPtr xla_data);

  Handle GetHandle() override;
  void Assign(const torch::lazy::BackendData& data) override;
  bool HasValue() const override;
  xla::ComputationClient::DataPtr xla_data();

 private:
  // TODO: Do we really need a Share_Ptr here?
  xla::ComputationClient::DataPtr xla_data_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_XLA_DATA_H