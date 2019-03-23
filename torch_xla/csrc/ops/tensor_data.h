#pragma once

#include <ATen/Tensor.h>

#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class TensorData : public Node {
 public:
  TensorData(at::Tensor tensor, Device device);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const at::Tensor& tensor() const { return tensor_; }

 private:
  static bool ShouldMakeConstant(const at::Tensor& tensor);

  static size_t GetTensorHashSeed(const at::Tensor& tensor);

  at::Tensor tensor_;
  Device device_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
