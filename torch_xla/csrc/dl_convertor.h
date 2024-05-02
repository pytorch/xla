#ifndef XLA_TORCH_XLA_CSRC_DL_CONVERTOR_H_
#define XLA_TORCH_XLA_CSRC_DL_CONVERTOR_H_

#include <ATen/dlpack.h>
#include <ATen/Tensor.h>

namespace torch_xla {

DLManagedTensor* toDLPack(const at::Tensor& src);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_DL_CONVERTOR_H_
