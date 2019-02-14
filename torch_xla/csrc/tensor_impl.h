#pragma once

#include <ATen/Tensor.h>
#include <ATen/Type.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an XLATensor.
class XLATensorImpl : public c10::TensorImpl {
 public:
  explicit XLATensorImpl(XLATensor tensor);
  XLATensorImpl(XLATensor tensor, bool is_variable);

  XLATensor& tensor() { return tensor_; }

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach() const override;

 private:
  void SetupSizeProperties();

  static caffe2::TypeMeta GetTypeMeta(const XLATensor& tensor);

  static c10::Storage GetStorage(const XLATensor& tensor);

  XLATensor tensor_;
};

}  // namespace torch_xla
