#pragma once

#include "aten_xla_type_base.h"

namespace torch_xla {

// Base ATEN Type class where the XLA specific overrides should be defined.
class AtenXlaType : public AtenXlaTypeBase {
 public:
  AtenXlaType(at::TensorTypeId type_id, bool is_variable, bool is_undefined);

  at::Tensor add(const at::Tensor& self, const at::Tensor& other,
                 at::Scalar alpha) const override;

  at::Tensor& add_(at::Tensor& self, const at::Tensor& other,
                   at::Scalar alpha) const override;

  at::Tensor mul(const at::Tensor& self,
                 const at::Tensor& other) const override;

  at::Tensor& mul_(at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor div(const at::Tensor& self,
                 const at::Tensor& other) const override;

  at::Tensor& div_(at::Tensor& self, const at::Tensor& other) const override;

  int64_t size(const at::Tensor& self, int64_t dim) const override;

  at::Tensor relu(const at::Tensor& self) const override;

  at::Tensor threshold(const at::Tensor& self, at::Scalar threshold,
                       at::Scalar value) const override;

  at::Tensor threshold_backward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                at::Scalar threshold) const override;

  at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight,
                    const at::Tensor& bias, at::IntList stride,
                    at::IntList padding, at::IntList dilation,
                    int64_t groups) const override;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> thnn_conv2d_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      const at::Tensor& weight, at::IntList kernel_size, at::IntList stride,
      at::IntList padding, const at::Tensor& finput,
      const at::Tensor& fgrad_input,
      std::array<bool, 3> output_mask) const override;

  at::Tensor addmm(const at::Tensor& self, const at::Tensor& mat1,
                   const at::Tensor& mat2, at::Scalar beta,
                   at::Scalar alpha) const override;

  at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) const override;

  at::Tensor t(const at::Tensor& self) const override;

  at::Tensor view(const at::Tensor& self, at::IntList size) const override;

  at::Tensor log_softmax(const at::Tensor& self, int64_t dim) const override;

  at::Tensor max_pool2d(const at::Tensor& self, at::IntList kernel_size,
                        at::IntList stride, at::IntList padding,
                        at::IntList dilation, bool ceil_mode) const override;

  at::Tensor avg_pool2d(const at::Tensor& self, at::IntList kernel_size,
                        at::IntList stride, at::IntList padding, bool ceil_mode,
                        bool count_include_pad) const override;

  at::Tensor avg_pool2d_backward(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 at::IntList kernel_size, at::IntList stride,
                                 at::IntList padding, bool ceil_mode,
                                 bool count_include_pad) const override;

  at::Tensor _log_softmax_backward_data(const at::Tensor& grad_output,
                                        const at::Tensor& output, int64_t dim,
                                        const at::Tensor& self) const override;

  static void SetFullConvPrecision(bool use_full_conv_precision = true);

  // Registers the ATEN types for the XLA tensors.
  static void RegisterAtenTypes();

 private:
  static bool s_use_full_conv_precision_;
};

}  // namespace torch_xla
