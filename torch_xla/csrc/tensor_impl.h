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
  explicit XLATensorImpl(XLATensor&& tensor);
  explicit XLATensorImpl(XLATensor& tensor);
  explicit XLATensorImpl(XLATensorPtr tensor);

  XLATensorPtr& tensor() { return tensor_; }

  void set_tensor(XLATensorPtr xla_tensor);

  void force_refresh_sizes() { generation_ = 0; }

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  at::IntArrayRef sizes_custom() const override;
  c10::SymIntArrayRef sym_sizes() const override;
  c10::SymIntArrayRef sym_sizes_custom() const override;
  at::IntArrayRef strides_custom() const override;

  int64_t dim_custom() const override;

  int64_t numel_custom() const override;

  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;

  const at::Storage& storage() const override;

  bool has_storage() const override;

  static void AtenInitialize();

 private:
  void SetupSizeProperties();

  static caffe2::TypeMeta GetTypeMeta(const XLATensor& tensor);

  XLATensorPtr tensor_;
  size_t generation_ = 0;
};

}  // namespace torch_xla
