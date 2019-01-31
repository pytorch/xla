#pragma once

#include <ATen/Tensor.h>
#include <ATen/Type.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "tensor.h"

namespace torch_xla {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an XLATensor.
class XLATensorImpl : public c10::TensorImpl {
 public:
  explicit XLATensorImpl(XLATensor tensor);
  XLATensorImpl(XLATensor tensor, bool is_variable);

  XLATensorImpl();

  XLATensor& tensor() { return tensor_; }

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach() const override;

 private:
  void SetupSizeProperties();

  static caffe2::TypeMeta GetTypeMeta(const XLATensor& tensor);

  static c10::Storage GetStorage(const XLATensor& tensor);

  XLATensor tensor_;
};

// The undefined counterpart to XLATensorImpl, needed for intrusive pointers.
class XLAUndefinedTensorImpl final : public XLATensorImpl {
 public:
  static constexpr inline XLATensorImpl* singleton() { return &_singleton; }
  at::IntList sizes() const override;
  at::IntList strides() const override;
  int64_t size(int64_t d) const override;
  int64_t stride(int64_t d) const override;
  int64_t dim() const override;
  const at::Storage& storage() const override;
  int64_t storage_offset() const override;

 private:
  XLAUndefinedTensorImpl();
  static XLAUndefinedTensorImpl _singleton;

 public:
  friend struct UndefinedType;
};

}  // namespace torch_xla
