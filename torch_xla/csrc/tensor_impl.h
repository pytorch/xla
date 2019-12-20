#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

// Tensor implementation class used to be fed to the at::Tensor.
// Its scope is just to handle an XLATensor.
class XLATensorImpl : public c10::TensorImpl {
 public:
  explicit XLATensorImpl(XLATensor tensor);

  XLATensor& tensor() { return tensor_; }

  void set_tensor(XLATensor xla_tensor);

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes() const override;

  int64_t dim() const override;

  int64_t numel() const override;

  bool is_contiguous(at::MemoryFormat memory_format) const override;

  int64_t size(int64_t d) const override;

  static void AtenInitialize();

  const at::Storage& storage() const override;

  bool has_storage() const override;

 private:
  void SetupSizeProperties();

  static caffe2::TypeMeta GetTypeMeta(const XLATensor& tensor);

  XLATensor tensor_;
  size_t generation_ = 0;
};

}  // namespace torch_xla
