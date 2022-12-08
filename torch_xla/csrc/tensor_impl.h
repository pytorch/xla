#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "torch/csrc/lazy/core/tensor_impl.h"
#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an XLATensor.
class XLATensorImpl : public torch::lazy::LTCTensorImpl {
 public:
  explicit XLATensorImpl(XLATensorPtr tensor);

  XLATensorPtr& tensor() { return tensor_; }

  void set_tensor(XLATensorPtr xla_tensor);

  at::IntArrayRef sizes_custom() const override;
  c10::SymIntArrayRef sym_sizes_custom() const override;
  c10::SymInt sym_numel_custom() const override;
  at::IntArrayRef strides_custom() const override;

  int64_t dim_custom() const override;

  int64_t numel_custom() const override;

  static void AtenInitialize();

 private:
  void SetupSizeProperties();
  void SetupSymSizeProperties();

  static caffe2::TypeMeta GetTypeMeta(const XLATensor& tensor);

  XLATensorPtr tensor_;
  std::vector<c10::SymInt> sym_sizes_;
  size_t generation_ = 0;
};

}  // namespace torch_xla
