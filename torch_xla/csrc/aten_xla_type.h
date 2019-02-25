#pragma once

#include "torch_xla/csrc/aten_xla_type_base.h"

namespace torch_xla {

// Base ATEN Type class where the XLA specific overrides should be defined.
class AtenXlaType : public AtenXlaTypeBase {
 public:
  AtenXlaType(at::TensorTypeId type_id, bool is_variable, bool is_undefined);

  int64_t numel(const at::Tensor& self) const override;

  at::Tensor _s_copy_from(const at::Tensor& self, const at::Tensor& dst,
                          bool non_blocking) const override;

  at::Tensor _cast_Byte(const at::Tensor& self,
                        bool non_blocking) const override;
  at::Tensor _cast_Char(const at::Tensor& self,
                        bool non_blocking) const override;
  at::Tensor _cast_Float(const at::Tensor& self,
                         bool non_blocking) const override;
  at::Tensor _cast_Int(const at::Tensor& self,
                       bool non_blocking) const override;
  at::Tensor _cast_Long(const at::Tensor& self,
                        bool non_blocking) const override;
  at::Tensor _cast_Short(const at::Tensor& self,
                         bool non_blocking) const override;

  at::Tensor& s_copy_(at::Tensor& self, const at::Tensor& src,
                      bool non_blocking) const override;

  at::Tensor contiguous(const at::Tensor& self) const override;

  at::Tensor empty(at::IntArrayRef size,
                   const at::TensorOptions& options) const override;
  at::Tensor empty_like(const at::Tensor& self) const override;
  at::Tensor empty_like(const at::Tensor& self,
                        const at::TensorOptions& options) const override;

  at::Tensor zeros(at::IntArrayRef size,
                   const at::TensorOptions& options) const override;
  at::Tensor zeros_like(const at::Tensor& self) const override;
  at::Tensor zeros_like(const at::Tensor& self,
                        const at::TensorOptions& options) const override;
  at::Tensor& zero_(at::Tensor& self) const override;

  at::Tensor ones(at::IntArrayRef size,
                  const at::TensorOptions& options) const override;
  at::Tensor ones_like(const at::Tensor& self) const override;
  at::Tensor ones_like(const at::Tensor& self,
                       const at::TensorOptions& options) const override;

  at::Tensor full(at::IntArrayRef size, at::Scalar fill_value,
                  const at::TensorOptions& options) const override;
  at::Tensor full_like(const at::Tensor& self,
                       at::Scalar fill_value) const override;
  at::Tensor full_like(const at::Tensor& self, at::Scalar fill_value,
                       const at::TensorOptions& options) const override;

  at::Tensor arange(at::Scalar end,
                    const at::TensorOptions& options) const override;
  at::Tensor arange(at::Scalar start, at::Scalar end,
                    const at::TensorOptions& options) const override;
  at::Tensor arange(at::Scalar start, at::Scalar end, at::Scalar step,
                    const at::TensorOptions& options) const override;
  at::Tensor _dim_arange(const at::Tensor& like, int64_t dim) const override;

  at::Tensor addcmul(const at::Tensor& self, const at::Tensor& tensor1,
                     const at::Tensor& tensor2,
                     at::Scalar value) const override;
  at::Tensor& addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                       const at::Tensor& tensor2,
                       at::Scalar value) const override;

  at::Tensor addcdiv(const at::Tensor& self, const at::Tensor& tensor1,
                     const at::Tensor& tensor2,
                     at::Scalar value) const override;
  at::Tensor& addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                       const at::Tensor& tensor2,
                       at::Scalar value) const override;

  at::Tensor exp(const at::Tensor& self) const override;

  at::Tensor expm1(const at::Tensor& self) const override;

  at::Tensor log(const at::Tensor& self) const override;

  at::Tensor log2(const at::Tensor& self) const override;

  at::Tensor log10(const at::Tensor& self) const override;

  at::Tensor log1p(const at::Tensor& self) const override;

  at::Tensor erf(const at::Tensor& self) const override;

  at::Tensor erfc(const at::Tensor& self) const override;

  at::Tensor erfinv(const at::Tensor& self) const override;

  at::Tensor sqrt(const at::Tensor& self) const override;

  at::Tensor rsqrt(const at::Tensor& self) const override;

  at::Tensor reciprocal(const at::Tensor& self) const override;

  at::Tensor pow(const at::Tensor& self, at::Scalar exponent) const override;

  at::Tensor add(const at::Tensor& self, const at::Tensor& other,
                 at::Scalar alpha) const override;
  at::Tensor& add_(at::Tensor& self, const at::Tensor& other,
                   at::Scalar alpha) const override;

  at::Tensor sub(const at::Tensor& self, const at::Tensor& other,
                 at::Scalar alpha) const override;
  at::Tensor& sub_(at::Tensor& self, const at::Tensor& other,
                   at::Scalar alpha) const override;

  at::Tensor rsub(const at::Tensor& self, const at::Tensor& other,
                  at::Scalar alpha) const override;

  at::Tensor rsub(const at::Tensor& self, at::Scalar other,
                  at::Scalar alpha) const override;

  at::Tensor mul(const at::Tensor& self,
                 const at::Tensor& other) const override;
  at::Tensor& mul_(at::Tensor& self, const at::Tensor& other) const override;
  at::Tensor& mul_(at::Tensor& self, at::Scalar other) const override;

  at::Tensor div(const at::Tensor& self,
                 const at::Tensor& other) const override;
  at::Tensor& div_(at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor fmod(const at::Tensor& self,
                  const at::Tensor& other) const override;
  at::Tensor& fmod_(at::Tensor& self, at::Scalar other) const override;
  at::Tensor& fmod_(at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor all(const at::Tensor& self) const override;
  at::Tensor all(const at::Tensor& self, int64_t dim,
                 bool keepdim) const override;
  at::Tensor any(const at::Tensor& self) const override;
  at::Tensor any(const at::Tensor& self, int64_t dim,
                 bool keepdim) const override;

  at::Tensor ne(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor ne(const at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor eq(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor eq(const at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor ge(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor ge(const at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor le(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor le(const at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor gt(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor gt(const at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor lt(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor lt(const at::Tensor& self, const at::Tensor& other) const override;

  at::Tensor __and__(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor __and__(const at::Tensor& self,
                     const at::Tensor& other) const override;

  at::Tensor __or__(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor __or__(const at::Tensor& self,
                    const at::Tensor& other) const override;

  at::Tensor __xor__(const at::Tensor& self, at::Scalar other) const override;

  at::Tensor __xor__(const at::Tensor& self,
                     const at::Tensor& other) const override;

  at::Tensor neg(const at::Tensor& self) const override;

  at::Tensor sign(const at::Tensor& self) const override;

  at::Tensor asin(const at::Tensor& self) const override;

  at::Tensor sin(const at::Tensor& self) const override;

  at::Tensor sinh(const at::Tensor& self) const override;

  at::Tensor acos(const at::Tensor& self) const override;

  at::Tensor cos(const at::Tensor& self) const override;

  at::Tensor cosh(const at::Tensor& self) const override;

  at::Tensor atan(const at::Tensor& self) const override;

  at::Tensor tan(const at::Tensor& self) const override;

  at::Tensor tanh(const at::Tensor& self) const override;

  at::Tensor abs(const at::Tensor& self) const override;

  at::Tensor sum(const at::Tensor& self, at::ScalarType dtype) const override;
  at::Tensor sum(const at::Tensor& self) const override;
  at::Tensor sum(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                 at::ScalarType dtype) const override;
  at::Tensor sum(const at::Tensor& self, at::IntArrayRef dim,
                 bool keepdim) const override;
  at::Tensor sum(const at::Tensor& self, at::IntArrayRef dim,
                 at::ScalarType dtype) const override;

  at::Tensor prod(const at::Tensor& self, at::ScalarType dtype) const override;
  at::Tensor prod(const at::Tensor& self) const override;
  at::Tensor prod(const at::Tensor& self, int64_t dim, bool keepdim,
                  at::ScalarType dtype) const override;
  at::Tensor prod(const at::Tensor& self, int64_t dim,
                  bool keepdim) const override;
  at::Tensor prod(const at::Tensor& self, int64_t dim,
                  at::ScalarType dtype) const override;

  at::Tensor clamp(const at::Tensor& self, c10::optional<at::Scalar> min,
                   c10::optional<at::Scalar> max) const override;

  at::Tensor& clamp_(at::Tensor& self, c10::optional<at::Scalar> min,
                     c10::optional<at::Scalar> max) const override;

  std::vector<at::Tensor> meshgrid(at::TensorList tensors) const override;

  at::Tensor constant_pad_nd(const at::Tensor& self, at::IntArrayRef pad,
                             at::Scalar value) const override;

  at::Tensor ceil(const at::Tensor& self) const override;

  at::Tensor floor(const at::Tensor& self) const override;

  int64_t size(const at::Tensor& self, int64_t dim) const override;

  at::Tensor embedding(const at::Tensor& weight, const at::Tensor& indices,
                       int64_t padding_idx, bool scale_grad_by_freq,
                       bool sparse) const override;

  at::Tensor slice(const at::Tensor& self, int64_t dim, int64_t start,
                   int64_t end, int64_t step) const override;

  at::Tensor gather(const at::Tensor& self, int64_t dim,
                    const at::Tensor& index) const override;

  at::Tensor index_select(const at::Tensor& self, int64_t dim,
                          const at::Tensor& index) const override;

  at::Tensor expand(const at::Tensor& self, at::IntArrayRef size,
                    bool implicit) const override;
  at::Tensor expand_as(const at::Tensor& self,
                       const at::Tensor& other) const override;

  at::Tensor stack(at::TensorList tensors, int64_t dim) const override;

  at::Tensor cat(at::TensorList tensors, int64_t dim) const override;

  std::vector<at::Tensor> unbind(const at::Tensor& self,
                                 int64_t dim) const override;

  at::Tensor relu(const at::Tensor& self) const override;
  at::Tensor& relu_(at::Tensor& self) const override;

  at::Tensor leaky_relu(const at::Tensor& self,
                        at::Scalar negative_slope) const override;

  at::Tensor threshold(const at::Tensor& self, at::Scalar threshold,
                       at::Scalar value) const override;

  at::Tensor threshold_backward(const at::Tensor& grad_output,
                                const at::Tensor& self,
                                at::Scalar threshold) const override;

  at::Tensor conv2d(const at::Tensor& input, const at::Tensor& weight,
                    const at::Tensor& bias, at::IntList stride,
                    at::IntList padding, at::IntList dilation,
                    int64_t groups) const override;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> thnn_conv2d_forward(
      const at::Tensor& self, const at::Tensor& weight,
      at::IntArrayRef kernel_size, const at::Tensor& bias,
      at::IntArrayRef stride, at::IntArrayRef padding) const override;

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

  at::Tensor matmul(const at::Tensor& self,
                    const at::Tensor& other) const override;

  at::Tensor bmm(const at::Tensor& self, const at::Tensor& mat2) const override;

  std::vector<at::Tensor> broadcast_tensors(
      at::TensorList tensors) const override;

  at::Tensor einsum(std::string equation,
                    at::TensorList tensors) const override;

  at::Tensor t(const at::Tensor& self) const override;

  at::Tensor reshape(const at::Tensor& self,
                     at::IntArrayRef shape) const override;

  at::Tensor view(const at::Tensor& self, at::IntList size) const override;

  at::Tensor select(const at::Tensor& self, int64_t dim,
                    int64_t index) const override;

  at::Tensor dropout(const at::Tensor& input, double p,
                     bool train) const override;

  at::Tensor log_softmax(const at::Tensor& self, int64_t dim) const override;
  at::Tensor _log_softmax(const at::Tensor& self, int64_t dim,
                          bool half_to_float) const override;

  at::Tensor softmax(const at::Tensor& self, int64_t dim) const override;

  at::Tensor sigmoid(const at::Tensor& self) const override;
  at::Tensor& sigmoid_(at::Tensor& self) const override;

  at::Tensor max_pool2d(const at::Tensor& self, at::IntList kernel_size,
                        at::IntList stride, at::IntList padding,
                        at::IntList dilation, bool ceil_mode) const override;

  std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(
      const at::Tensor& self, at::IntList kernel_size, at::IntList stride,
      at::IntList padding, at::IntList dilation, bool ceil_mode) const override;

  at::Tensor avg_pool2d(const at::Tensor& self, at::IntList kernel_size,
                        at::IntList stride, at::IntList padding, bool ceil_mode,
                        bool count_include_pad) const override;

  at::Tensor _adaptive_avg_pool2d(const at::Tensor& self,
                                  at::IntArrayRef output_size) const override;

  at::Tensor batch_norm(const at::Tensor& input, const at::Tensor& weight,
                        const at::Tensor& bias, const at::Tensor& running_mean,
                        const at::Tensor& running_var, bool training,
                        double momentum, double eps,
                        bool cudnn_enabled) const override;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
      const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
      const at::Tensor& running_mean, const at::Tensor& running_var,
      bool training, double momentum, double eps) const override;

  at::Tensor avg_pool2d_backward(const at::Tensor& grad_output,
                                 const at::Tensor& self,
                                 at::IntList kernel_size, at::IntList stride,
                                 at::IntList padding, bool ceil_mode,
                                 bool count_include_pad) const override;

  at::Tensor _adaptive_avg_pool2d_backward(
      const at::Tensor& grad_output, const at::Tensor& self) const override;

  at::Tensor max_pool2d_with_indices_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      at::IntList kernel_size, at::IntList stride, at::IntList padding,
      at::IntList dilation, bool ceil_mode,
      const at::Tensor& indices) const override;

  at::Tensor _log_softmax_backward_data(const at::Tensor& grad_output,
                                        const at::Tensor& output, int64_t dim,
                                        const at::Tensor& self) const override;

  at::Tensor nll_loss(const at::Tensor& self, const at::Tensor& target,
                      const at::Tensor& weight, int64_t reduction,
                      int64_t ignore_index) const override;
  std::tuple<at::Tensor, at::Tensor> nll_loss_forward(
      const at::Tensor& self, const at::Tensor& target,
      const at::Tensor& weight, int64_t reduction,
      int64_t ignore_index) const override;

  at::Tensor nll_loss_backward(const at::Tensor& grad_output,
                               const at::Tensor& self, const at::Tensor& target,
                               const at::Tensor& weight, int64_t reduction,
                               int64_t ignore_index,
                               const at::Tensor& total_weight) const override;

  at::Tensor min(const at::Tensor& self,
                 const at::Tensor& other) const override;
  at::Tensor max(const at::Tensor& self,
                 const at::Tensor& other) const override;

  at::Tensor mean(const at::Tensor& self, at::ScalarType dtype) const override;
  at::Tensor mean(const at::Tensor& self) const override;
  at::Tensor mean(const at::Tensor& self, at::IntArrayRef dim, bool keepdim,
                  at::ScalarType dtype) const override;
  at::Tensor mean(const at::Tensor& self, at::IntArrayRef dim,
                  bool keepdim) const override;
  at::Tensor mean(const at::Tensor& self, at::IntArrayRef dim,
                  at::ScalarType dtype) const override;

  at::Tensor argmax(const at::Tensor& self, int64_t dim,
                    bool keepdim) const override;
  at::Tensor argmax(const at::Tensor& self) const override;

  at::Tensor argmin(const at::Tensor& self, int64_t dim,
                    bool keepdim) const override;
  at::Tensor argmin(const at::Tensor& self) const override;

  std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm_backward(
      const at::Tensor& grad_out, const at::Tensor& input,
      const at::Tensor& weight, const at::Tensor& running_mean,
      const at::Tensor& running_var, const at::Tensor& save_mean,
      const at::Tensor& save_invstd, bool train, double eps,
      std::array<bool, 3> output_mask) const override;

  at::Tensor permute(const at::Tensor& self,
                     at::IntArrayRef dims) const override;

  at::Tensor repeat(const at::Tensor& self,
                    at::IntArrayRef repeats) const override;

  std::vector<at::Tensor> split(const at::Tensor& self, int64_t split_size,
                                int64_t dim) const override;
  std::vector<at::Tensor> split_with_sizes(const at::Tensor& self,
                                           at::IntArrayRef split_sizes,
                                           int64_t dim) const override;

  at::Tensor squeeze(const at::Tensor& self) const override;
  at::Tensor squeeze(const at::Tensor& self, int64_t dim) const override;

  at::Tensor& squeeze_(at::Tensor& self) const override;
  at::Tensor& squeeze_(at::Tensor& self, int64_t dim) const override;

  at::Tensor unsqueeze(const at::Tensor& self, int64_t dim) const override;

  at::Tensor where(const at::Tensor& condition, const at::Tensor& self,
                   const at::Tensor& other) const override;

  at::Tensor triu(const at::Tensor& self, int64_t diagonal) const override;

  at::Tensor tril(const at::Tensor& self, int64_t diagonal) const override;

  at::Tensor diagonal(const at::Tensor& self, int64_t offset, int64_t dim1,
                      int64_t dim2) const override;

  // Registers the ATEN types for the XLA tensors.
  static void RegisterAtenTypes();
};

}  // namespace torch_xla
