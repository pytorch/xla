#ifndef XLA_TORCH_XLA_CSRC_TENSOR_IMPL_H_
#define XLA_TORCH_XLA_CSRC_TENSOR_IMPL_H_

#include <ATen/Tensor.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>

#include "torch_xla/csrc/tensor.h"

namespace torch_xla {

// [Note: Re-using upstream TensorImpl] As part of the LTC migration effort,
// we tried to re-use the upstream LTCTensorImpl
// (torch/csrc/lazy/core/tensor_impl.h) instead of having our own version of
// XLATensorImpl. However, LTCTensorImpl defines its own set of hard-coded
// dispatch keys in its constructor and has functions that call this
// constructor. In addition, updating XLATensorImpl to extend LTCTensorImpl does
// not produce much benefit since that wouldn't remove much duplicate code. As a
// result, we decided to keep and use XLATensorImpl.

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
  c10::SymIntArrayRef sym_sizes_custom() const override;
  c10::SymInt sym_numel_custom() const override;
  at::IntArrayRef strides_custom() const override;

  int64_t dim_custom() const override;

  int64_t numel_custom() const override;

  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;

  const at::Storage& storage() const override;

  bool has_storage() const override;

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

#endif  // XLA_TORCH_XLA_CSRC_TENSOR_IMPL_H_