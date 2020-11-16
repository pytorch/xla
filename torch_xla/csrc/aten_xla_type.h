#pragma once

#include <ATen/Tensor.h>

namespace torch_xla {

// Base ATEN Type class where the XLA specific overrides should be defined.
class AtenXlaType {
 public:
  static void InitializeAtenBindings();

  //////////////////////////////////////////////////////////////////////////////
  // ATEN API ovverrides in alphabetical order.
  // Note: The C++ signatures must match the ones listed within the following
  // pytorch folder file:
  //   torch/csrc/autograd/generated/RegistrationDeclarations.h
  /////////////////////////////////////////////////////////////////////////////
  static at::Tensor& __ilshift__(at::Tensor& self, at::Scalar other);

  static at::Tensor& __ilshift__(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& __irshift__(at::Tensor& self, at::Scalar other);

  static at::Tensor& __irshift__(at::Tensor& self, const at::Tensor& other);

  static at::Tensor __lshift__(const at::Tensor& self, at::Scalar other);

  static at::Tensor __lshift__(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor __rshift__(const at::Tensor& self, at::Scalar other);

  static at::Tensor __rshift__(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor _adaptive_avg_pool2d(const at::Tensor& self,
                                         at::IntArrayRef output_size);

  static at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor& grad_output,
                                                  const at::Tensor& self);

  static at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst,
                               bool non_blocking);

  static at::Tensor& _index_put_impl_(at::Tensor& self, at::TensorList indices,
                                      const at::Tensor& values, bool accumulate,
                                      bool unsafe);

  static at::Tensor _log_softmax(const at::Tensor& self, int64_t dim,
                                 bool half_to_float);

  static at::Tensor _log_softmax_backward_data(const at::Tensor& grad_output,
                                               const at::Tensor& output,
                                               int64_t dim,
                                               const at::Tensor& self);

  static std::tuple<at::Tensor, at::Tensor> _pack_padded_sequence(
      const at::Tensor& input, const at::Tensor& lengths, bool batch_first);

  static at::Tensor _s_where(const at::Tensor& condition,
                             const at::Tensor& self, const at::Tensor& other);

  static at::Tensor _softmax(const at::Tensor& self, int64_t dim,
                             bool half_to_float);

  static at::Tensor _softmax_backward_data(const at::Tensor& grad_output,
                                           const at::Tensor& output,
                                           int64_t dim, const at::Tensor& self);

  static at::Tensor _trilinear(const at::Tensor& i1, const at::Tensor& i2,
                               const at::Tensor& i3, at::IntArrayRef expand1,
                               at::IntArrayRef expand2, at::IntArrayRef expand3,
                               at::IntArrayRef sumdim, int64_t unroll_dim);

  static at::Tensor _unsafe_view(const at::Tensor& self, at::IntArrayRef size);

  static at::Tensor abs(const at::Tensor& self);

  static at::Tensor& abs_(at::Tensor& self);

  static at::Tensor acos(const at::Tensor& self);

  static at::Tensor& acos_(at::Tensor& self);

  static at::Tensor acosh(const at::Tensor& self);

  static at::Tensor& acosh_(at::Tensor& self);

  static at::Tensor add(const at::Tensor& self, const at::Tensor& other,
                        at::Scalar alpha);

  static at::Tensor add(const at::Tensor& self, at::Scalar other,
                        at::Scalar alpha);

  static at::Tensor& add_(at::Tensor& self, const at::Tensor& other,
                          at::Scalar alpha);

  static at::Tensor& add_(at::Tensor& self, at::Scalar other, at::Scalar alpha);

  static at::Tensor addcdiv(const at::Tensor& self, const at::Tensor& tensor1,
                            const at::Tensor& tensor2, at::Scalar value);

  static at::Tensor& addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                              const at::Tensor& tensor2, at::Scalar value);

  static at::Tensor addcmul(const at::Tensor& self, const at::Tensor& tensor1,
                            const at::Tensor& tensor2, at::Scalar value);

  static at::Tensor& addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                              const at::Tensor& tensor2, at::Scalar value);

  static at::Tensor addmm(const at::Tensor& self, const at::Tensor& mat1,
                          const at::Tensor& mat2, at::Scalar beta,
                          at::Scalar alpha);

  static at::Tensor alias(const at::Tensor& self);

  static at::Tensor all(const at::Tensor& self);

  static at::Tensor all(const at::Tensor& self, int64_t dim, bool keepdim);

  static at::Tensor any(const at::Tensor& self);

  static at::Tensor any(const at::Tensor& self, int64_t dim, bool keepdim);

  static at::Tensor& arange_out(at::Tensor& out, at::Scalar start,
                                at::Scalar end, at::Scalar step);

  static at::Tensor argmax(const at::Tensor& self, c10::optional<int64_t> dim,
                           bool keepdim);

  static at::Tensor argmin(const at::Tensor& self, c10::optional<int64_t> dim,
                           bool keepdim);

  static at::Tensor as_strided(const at::Tensor& self, at::IntArrayRef size,
                               at::IntArrayRef stride,
                               c10::optional<int64_t> storage_offset);

  static at::Tensor& as_strided_(at::Tensor& self, at::IntArrayRef size,
                                 at::IntArrayRef stride,
                                 c10::optional<int64_t> storage_offset);

  static at::Tensor asin(const at::Tensor& self);

  static at::Tensor& asin_(at::Tensor& self);

  static at::Tensor asinh(const at::Tensor& self);

  static at::Tensor& asinh_(at::Tensor& self);

  static at::Tensor atan(const at::Tensor& self);

  static at::Tensor atanh(const at::Tensor& self);

  static at::Tensor atan2(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& atan2_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& atan_(at::Tensor& self);

  static at::Tensor& atanh_(at::Tensor& self);

  static at::Tensor avg_pool2d(const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               bool ceil_mode, bool count_include_pad,
                               c10::optional<int64_t> divisor_override);

  static at::Tensor avg_pool2d_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      at::IntArrayRef kernel_size, at::IntArrayRef stride,
      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
      c10::optional<int64_t> divisor_override);

  static at::Tensor avg_pool3d(const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               bool ceil_mode, bool count_include_pad,
                               c10::optional<int64_t> divisor_override);

  static at::Tensor avg_pool3d_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      at::IntArrayRef kernel_size, at::IntArrayRef stride,
      at::IntArrayRef padding, bool ceil_mode, bool count_include_pad,
      c10::optional<int64_t> divisor_override);

  static at::Tensor baddbmm(const at::Tensor& self, const at::Tensor& batch1,
                            const at::Tensor& batch2, at::Scalar beta,
                            at::Scalar alpha);
  static at::Tensor& baddbmm_(at::Tensor& self, const at::Tensor& batch1,
                              const at::Tensor& batch2, at::Scalar beta,
                              at::Scalar alpha);

  static at::Tensor bernoulli(const at::Tensor& self,
                              c10::optional<at::Generator> generator);
  static at::Tensor& bernoulli_(at::Tensor& self, double p,
                                c10::optional<at::Generator> generator);
  static at::Tensor& bernoulli_(at::Tensor& self, const at::Tensor& p,
                                c10::optional<at::Generator> generator);

  // binary_cross_entropy is still in PyTorch TH legacy, which means both are
  // overrideable for XLA. But overriding one set is sufficient. Currently
  // binary_cross_entropy and its backward are used instead of the _out version.
  // When it's moved to Aten in the future, we should only keep one set here.
  static at::Tensor binary_cross_entropy(
      const at::Tensor& self, const at::Tensor& target,
      const c10::optional<at::Tensor>& weight, int64_t reduction);
  static at::Tensor binary_cross_entropy_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      const at::Tensor& target, const c10::optional<at::Tensor>& weight,
      int64_t reduction);

  static at::Tensor binary_cross_entropy_with_logits(
      const at::Tensor& self, const at::Tensor& target,
      const c10::optional<at::Tensor>& weight,
      const c10::optional<at::Tensor>& pos_weight, int64_t reduction);

  static at::Tensor& bitwise_and_out(at::Tensor& out, const at::Tensor& self,
                                     const at::Tensor& other);

  static at::Tensor& bitwise_and_out(at::Tensor& out, const at::Tensor& self,
                                     at::Scalar other);

  static at::Tensor& bitwise_not_out(at::Tensor& out, const at::Tensor& self);

  static at::Tensor& bitwise_or_out(at::Tensor& out, const at::Tensor& self,
                                    const at::Tensor& other);

  static at::Tensor& bitwise_or_out(at::Tensor& out, const at::Tensor& self,
                                    at::Scalar other);

  static at::Tensor& bitwise_xor_out(at::Tensor& out, const at::Tensor& self,
                                     at::Scalar other);

  static at::Tensor& bitwise_xor_out(at::Tensor& out, const at::Tensor& self,
                                     const at::Tensor& other);

  static at::Tensor bmm(const at::Tensor& self, const at::Tensor& mat2);

  static at::Tensor cat(at::TensorList tensors, int64_t dim);

  static at::Tensor ceil(const at::Tensor& self);

  static at::Tensor& ceil_(at::Tensor& self);

  static at::Tensor cholesky(const at::Tensor& self, bool upper);

  static at::Tensor clamp(const at::Tensor& self, c10::optional<at::Scalar> min,
                          c10::optional<at::Scalar> max);

  static at::Tensor& clamp_(at::Tensor& self, c10::optional<at::Scalar> min,
                            c10::optional<at::Scalar> max);

  static at::Tensor clamp_max(const at::Tensor& self, at::Scalar max);

  static at::Tensor& clamp_max_(at::Tensor& self, at::Scalar max);

  static at::Tensor clamp_min(const at::Tensor& self, at::Scalar min);

  static at::Tensor& clamp_min_(at::Tensor& self, at::Scalar min);

  static at::Tensor clone(const at::Tensor& self,
                          c10::optional<at::MemoryFormat> memory_format);

  static at::Tensor constant_pad_nd(const at::Tensor& self, at::IntArrayRef pad,
                                    at::Scalar value);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor>
  convolution_backward_overrideable(
      const at::Tensor& grad_output, const at::Tensor& input,
      const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
      at::IntArrayRef dilation, bool transposed, at::IntArrayRef output_padding,
      int64_t groups, std::array<bool, 3> output_mask);

  static at::Tensor convolution_overrideable(
      const at::Tensor& input, const at::Tensor& weight,
      const c10::optional<at::Tensor>& bias, at::IntArrayRef stride,
      at::IntArrayRef padding, at::IntArrayRef dilation, bool transposed,
      at::IntArrayRef output_padding, int64_t groups);

  static at::Tensor cos(const at::Tensor& self);

  static at::Tensor& cos_(at::Tensor& self);

  static at::Tensor cosh(const at::Tensor& self);

  static at::Tensor& cosh_(at::Tensor& self);

  static at::Tensor cross(const at::Tensor& self, const at::Tensor& other,
                          c10::optional<int64_t> dim);

  static at::Tensor cumprod(const at::Tensor& self, int64_t dim,
                            c10::optional<at::ScalarType> dtype);

  static at::Tensor cumsum(const at::Tensor& self, int64_t dim,
                           c10::optional<at::ScalarType> dtype);

  static at::Tensor diag(const at::Tensor& self, int64_t diagonal);

  static at::Tensor diagonal(const at::Tensor& self, int64_t offset,
                             int64_t dim1, int64_t dim2);

  static at::Tensor div(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor div(const at::Tensor& self, at::Scalar other);

  static at::Tensor& div_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& div_(at::Tensor& self, at::Scalar other);

  static at::Tensor dot(const at::Tensor& self, const at::Tensor& tensor);

  static at::Tensor elu(const at::Tensor& self, at::Scalar alpha,
                        at::Scalar scale, at::Scalar input_scale);

  static at::Tensor& elu_(at::Tensor& self, at::Scalar alpha, at::Scalar scale,
                          at::Scalar input_scale);

  static at::Tensor elu_backward(const at::Tensor& grad_output,
                                 at::Scalar alpha, at::Scalar scale,
                                 at::Scalar input_scale,
                                 const at::Tensor& output);

  static at::Tensor embedding(const at::Tensor& weight,
                              const at::Tensor& indices, int64_t padding_idx,
                              bool scale_grad_by_freq, bool sparse);

  static at::Tensor embedding_dense_backward(const at::Tensor& grad_output,
                                             const at::Tensor& indices,
                                             int64_t num_weights,
                                             int64_t padding_idx,
                                             bool scale_grad_by_freq);

  static at::Tensor empty(at::IntArrayRef size,
                          c10::optional<at::ScalarType> dtype,
                          c10::optional<at::Layout> layout,
                          c10::optional<at::Device> device,
                          c10::optional<bool> pin_memory,
                          c10::optional<at::MemoryFormat> memory_format);

  static at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                                  c10::optional<at::ScalarType> dtype,
                                  c10::optional<at::Layout> layout,
                                  c10::optional<at::Device> device,
                                  c10::optional<bool> pin_memory);

  static at::Tensor eq(const at::Tensor& self, at::Scalar other);

  static at::Tensor eq(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& eq_(at::Tensor& self, at::Scalar other);

  static at::Tensor& eq_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor erf(const at::Tensor& self);

  static at::Tensor& erf_(at::Tensor& self);

  static at::Tensor erfc(const at::Tensor& self);

  static at::Tensor& erfc_(at::Tensor& self);

  static at::Tensor erfinv(const at::Tensor& self);

  static at::Tensor& erfinv_(at::Tensor& self);

  static at::Tensor exp(const at::Tensor& self);

  static at::Tensor& exp_(at::Tensor& self);

  static at::Tensor expand(const at::Tensor& self, at::IntArrayRef size,
                           bool implicit);

  static at::Tensor expm1(const at::Tensor& self);

  static at::Tensor& expm1_(at::Tensor& self);

  static at::Tensor& exponential_(at::Tensor& self, double lambd,
                                  c10::optional<at::Generator> generator);

  static at::Tensor& eye_out(at::Tensor& out, int64_t n);

  static at::Tensor& eye_out(at::Tensor& out, int64_t n, int64_t m);

  static at::Tensor& fill_(at::Tensor& self, at::Scalar value);

  static at::Tensor& fill_(at::Tensor& self, const at::Tensor& value);

  static at::Tensor flip(const at::Tensor& self, at::IntArrayRef dims);

  static at::Tensor floor(const at::Tensor& self);

  static at::Tensor& floor_(at::Tensor& self);

  static at::Tensor fmod(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor fmod(const at::Tensor& self, at::Scalar other);

  static at::Tensor& fmod_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& fmod_(at::Tensor& self, at::Scalar other);

  static at::Tensor frac(const at::Tensor& self);

  static at::Tensor& frac_(at::Tensor& self);

  static at::Tensor gather(const at::Tensor& self, int64_t dim,
                           const at::Tensor& index, bool sparse_grad);

  static at::Tensor ge(const at::Tensor& self, at::Scalar other);

  static at::Tensor ge(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& ge_(at::Tensor& self, at::Scalar other);

  static at::Tensor& ge_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor gelu(const at::Tensor& self);

  static at::Tensor gelu_backward(const at::Tensor& grad,
                                  const at::Tensor& self);

  static at::Tensor ger(const at::Tensor& self, const at::Tensor& vec2);

  static at::Tensor gt(const at::Tensor& self, at::Scalar other);

  static at::Tensor gt(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& gt_(at::Tensor& self, at::Scalar other);

  static at::Tensor& gt_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor hardshrink(const at::Tensor& self, at::Scalar lambda);

  static at::Tensor hardshrink_backward(const at::Tensor& grad_out,
                                        const at::Tensor& self,
                                        at::Scalar lambda);

  static at::Tensor hardsigmoid(const at::Tensor& self);

  static at::Tensor& hardsigmoid_(at::Tensor& self);

  static at::Tensor hardsigmoid_backward(const at::Tensor& grad_output,
                                         const at::Tensor& self);

  static at::Tensor hardtanh(const at::Tensor& self, at::Scalar min_val,
                             at::Scalar max_val);

  static at::Tensor& hardtanh_(at::Tensor& self, at::Scalar min_val,
                               at::Scalar max_val);

  static at::Tensor hardtanh_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      at::Scalar min_val, at::Scalar max_val);

  static at::Tensor index(const at::Tensor& self, at::TensorList indices);

  static at::Tensor& index_add_(at::Tensor& self, int64_t dim,
                                const at::Tensor& index,
                                const at::Tensor& source);

  static at::Tensor& index_copy_(at::Tensor& self, int64_t dim,
                                 const at::Tensor& index,
                                 const at::Tensor& source);

  static at::Tensor& index_fill_(at::Tensor& self, int64_t dim,
                                 const at::Tensor& index, at::Scalar value);

  static at::Tensor& index_fill_(at::Tensor& self, int64_t dim,
                                 const at::Tensor& index,
                                 const at::Tensor& value);

  static at::Tensor& index_put_(at::Tensor& self, at::TensorList indices,
                                const at::Tensor& values, bool accumulate);

  static at::Tensor index_select(const at::Tensor& self, int64_t dim,
                                 const at::Tensor& index);

  static at::Tensor inverse(const at::Tensor& self);

  static at::Tensor kl_div(const at::Tensor& self, const at::Tensor& target,
                           int64_t reduction, bool log_target);

  static at::Tensor kl_div_backward(const at::Tensor& grad_output,
                                    const at::Tensor& self,
                                    const at::Tensor& target, int64_t reduction,
                                    bool log_target);

  static std::tuple<at::Tensor, at::Tensor> kthvalue(const at::Tensor& self,
                                                     int64_t k, int64_t dim,
                                                     bool keepdim);

  static at::Tensor l1_loss(const at::Tensor& self, const at::Tensor& target,
                            int64_t reduction);

  static at::Tensor l1_loss_backward(const at::Tensor& grad_output,
                                     const at::Tensor& self,
                                     const at::Tensor& target,
                                     int64_t reduction);

  static at::Tensor le(const at::Tensor& self, at::Scalar other);

  static at::Tensor le(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& le_(at::Tensor& self, at::Scalar other);

  static at::Tensor& le_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor leaky_relu(const at::Tensor& self,
                               at::Scalar negative_slope);

  static at::Tensor& leaky_relu_(at::Tensor& self, at::Scalar negative_slope);

  static at::Tensor leaky_relu_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        at::Scalar negative_slope,
                                        bool self_is_result);

  static at::Tensor log(const at::Tensor& self);

  static at::Tensor log10(const at::Tensor& self);

  static at::Tensor& log10_(at::Tensor& self);

  static at::Tensor log1p(const at::Tensor& self);

  static at::Tensor& log1p_(at::Tensor& self);

  static at::Tensor log2(const at::Tensor& self);

  static at::Tensor& log2_(at::Tensor& self);

  static at::Tensor& log_(at::Tensor& self);

  static at::Tensor log_sigmoid_backward(const at::Tensor& grad_output,
                                         const at::Tensor& self,
                                         const at::Tensor& buffer);

  static std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward(
      const at::Tensor& self);

  static at::Tensor logdet(const at::Tensor& self);

  static at::Tensor logsumexp(const at::Tensor& self, at::IntArrayRef dim,
                              bool keepdim);

  static at::Tensor lt(const at::Tensor& self, at::Scalar other);

  static at::Tensor lt(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& lt_(at::Tensor& self, at::Scalar other);

  static at::Tensor& lt_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                  at::Scalar value);

  static at::Tensor& masked_fill_(at::Tensor& self, const at::Tensor& mask,
                                  const at::Tensor& value);

  static at::Tensor& masked_scatter_(at::Tensor& self, const at::Tensor& mask,
                                     const at::Tensor& source);

  static at::Tensor masked_select(const at::Tensor& self,
                                  const at::Tensor& mask);

  static at::Tensor max(const at::Tensor& self);

  static std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self,
                                                int64_t dim, bool keepdim);

  static at::Tensor maximum(const at::Tensor& self, const at::Tensor& other);

  static std::tuple<at::Tensor&, at::Tensor&> max_out(at::Tensor& max,
                                                      at::Tensor& max_values,
                                                      const at::Tensor& self,
                                                      int64_t dim,
                                                      bool keepdim);

  static at::Tensor max_pool2d(const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool ceil_mode);

  static std::tuple<at::Tensor, at::Tensor> max_pool2d_with_indices(
      const at::Tensor& self, at::IntArrayRef kernel_size,
      at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
      bool ceil_mode);

  static at::Tensor max_pool2d_with_indices_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      at::IntArrayRef kernel_size, at::IntArrayRef stride,
      at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
      const at::Tensor& indices);

  static at::Tensor max_pool3d(const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool ceil_mode);

  static std::tuple<at::Tensor, at::Tensor> max_pool3d_with_indices(
      const at::Tensor& self, at::IntArrayRef kernel_size,
      at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
      bool ceil_mode);

  static at::Tensor max_pool3d_with_indices_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      at::IntArrayRef kernel_size, at::IntArrayRef stride,
      at::IntArrayRef padding, at::IntArrayRef dilation, bool ceil_mode,
      const at::Tensor& indices);

  static at::Tensor max_unpool2d(const at::Tensor& self,
                                 const at::Tensor& indices,
                                 at::IntArrayRef output_size);

  static at::Tensor max_unpool2d_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Tensor& indices,
                                          at::IntArrayRef output_size);

  static at::Tensor max_unpool3d(const at::Tensor& self,
                                 const at::Tensor& indices,
                                 at::IntArrayRef output_size,
                                 at::IntArrayRef stride,
                                 at::IntArrayRef padding);

  static at::Tensor max_unpool3d_backward(const at::Tensor& grad_output,
                                          const at::Tensor& self,
                                          const at::Tensor& indices,
                                          at::IntArrayRef output_size,
                                          at::IntArrayRef stride,
                                          at::IntArrayRef padding);

  static at::Tensor mean(const at::Tensor& self,
                         c10::optional<at::ScalarType> dtype);

  static at::Tensor mean(const at::Tensor& self, at::IntArrayRef dim,
                         bool keepdim, c10::optional<at::ScalarType> dtype);

  static at::Tensor min(const at::Tensor& self);

  static std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self,
                                                int64_t dim, bool keepdim);

  static at::Tensor minimum(const at::Tensor& self, const at::Tensor& other);

  static std::tuple<at::Tensor&, at::Tensor&> min_out(at::Tensor& min,
                                                      at::Tensor& min_indices,
                                                      const at::Tensor& self,
                                                      int64_t dim,
                                                      bool keepdim);

  static at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2);

  static at::Tensor mse_loss(const at::Tensor& self, const at::Tensor& target,
                             int64_t reduction);

  static at::Tensor mse_loss_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      const at::Tensor& target,
                                      int64_t reduction);

  static at::Tensor mul(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor mul(const at::Tensor& self, at::Scalar other);

  static at::Tensor& mul_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& mul_(at::Tensor& self, at::Scalar other);

  static at::Tensor mv(const at::Tensor& self, const at::Tensor& vec);

  static at::Tensor& mv_out(at::Tensor& out, const at::Tensor& self,
                            const at::Tensor& vec);

  static at::Tensor narrow_copy(const at::Tensor& self, int64_t dim,
                                int64_t start, int64_t length);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor> native_batch_norm(
      const at::Tensor& input, const c10::optional<at::Tensor>& weight,
      const c10::optional<at::Tensor>& bias,
      const c10::optional<at::Tensor>& running_mean,
      const c10::optional<at::Tensor>& running_var, bool training,
      double momentum, double eps);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor>
  native_batch_norm_backward(const at::Tensor& grad_out,
                             const at::Tensor& input,
                             const c10::optional<at::Tensor>& weight,
                             const c10::optional<at::Tensor>& running_mean,
                             const c10::optional<at::Tensor>& running_var,
                             const c10::optional<at::Tensor>& save_mean,
                             const c10::optional<at::Tensor>& save_invstd,
                             bool train, double eps,
                             std::array<bool, 3> output_mask);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor>
  native_group_norm_backward(const at::Tensor& grad_out,
                             const at::Tensor& input, const at::Tensor& mean,
                             const at::Tensor& rstd,
                             const c10::optional<at::Tensor>& weight, int64_t N,
                             int64_t C, int64_t HxW, int64_t group,
                             std::array<bool, 3> output_mask);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor>
  native_layer_norm_backward(const at::Tensor& grad_out,
                             const at::Tensor& input, const at::Tensor& mean,
                             const at::Tensor& rstd,
                             const c10::optional<at::Tensor>& weight, int64_t M,
                             int64_t N, std::array<bool, 3> output_mask);

  static at::Tensor ne(const at::Tensor& self, at::Scalar other);

  static at::Tensor ne(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& ne_(at::Tensor& self, at::Scalar other);

  static at::Tensor& ne_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor neg(const at::Tensor& self);

  static at::Tensor& neg_(at::Tensor& self);

  static at::Tensor nll_loss2d_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Tensor& target,
                                        const c10::optional<at::Tensor>& weight,
                                        int64_t reduction, int64_t ignore_index,
                                        const at::Tensor& total_weight);

  static std::tuple<at::Tensor, at::Tensor> nll_loss2d_forward(
      const at::Tensor& self, const at::Tensor& target,
      const c10::optional<at::Tensor>& weight, int64_t reduction,
      int64_t ignore_index);

  static at::Tensor nll_loss_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      const at::Tensor& target,
                                      const c10::optional<at::Tensor>& weight,
                                      int64_t reduction, int64_t ignore_index,
                                      const at::Tensor& total_weight);

  static std::tuple<at::Tensor, at::Tensor> nll_loss_forward(
      const at::Tensor& self, const at::Tensor& target,
      const c10::optional<at::Tensor>& weight, int64_t reduction,
      int64_t ignore_index);

  static at::Tensor nonzero(const at::Tensor& self);

  static at::Tensor norm(const at::Tensor& self, c10::optional<at::Scalar> p,
                         at::ScalarType dtype);

  static at::Tensor norm(const at::Tensor& self, at::Scalar p);

  static at::Tensor norm(const at::Tensor& self, c10::optional<at::Scalar> p,
                         at::IntArrayRef dim, bool keepdim,
                         at::ScalarType dtype);

  static at::Tensor norm(const at::Tensor& self, c10::optional<at::Scalar> p,
                         at::IntArrayRef dim, bool keepdim);

  static at::Tensor normal(const at::Tensor& mean, double std,
                           c10::optional<at::Generator> generator);

  static at::Tensor normal(double mean, const at::Tensor& std,
                           c10::optional<at::Generator> generator);

  static at::Tensor normal(const at::Tensor& mean, const at::Tensor& std,
                           c10::optional<at::Generator> generator);

  static at::Tensor& normal_(at::Tensor& self, double mean, double std,
                             c10::optional<at::Generator> generator);

  static at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims);

  static at::Tensor pow(const at::Tensor& self, at::Scalar exponent);

  static at::Tensor pow(const at::Tensor& self, const at::Tensor& exponent);

  static at::Tensor pow(at::Scalar self, const at::Tensor& exponent);

  static at::Tensor& pow_(at::Tensor& self, at::Scalar exponent);

  static at::Tensor& pow_(at::Tensor& self, const at::Tensor& exponent);

  static at::Tensor prod(const at::Tensor& self,
                         c10::optional<at::ScalarType> dtype);

  static at::Tensor prod(const at::Tensor& self, int64_t dim, bool keepdim,
                         c10::optional<at::ScalarType> dtype);

  static at::Tensor& put_(at::Tensor& self, const at::Tensor& index,
                          const at::Tensor& source, bool accumulate);

  static std::tuple<at::Tensor, at::Tensor> qr(const at::Tensor& self,
                                               bool some);

  static at::Tensor reciprocal(const at::Tensor& self);

  static at::Tensor& reciprocal_(at::Tensor& self);

  static at::Tensor reflection_pad2d(const at::Tensor& self,
                                     at::IntArrayRef padding);

  static at::Tensor reflection_pad2d_backward(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              at::IntArrayRef padding);

  static at::Tensor relu(const at::Tensor& self);

  static at::Tensor& relu_(at::Tensor& self);

  static at::Tensor remainder(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor remainder(const at::Tensor& self, at::Scalar other);

  static at::Tensor& remainder_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& remainder_(at::Tensor& self, at::Scalar other);

  static at::Tensor repeat(const at::Tensor& self, at::IntArrayRef repeats);

  static at::Tensor replication_pad1d(const at::Tensor& self,
                                      at::IntArrayRef padding);
  static at::Tensor replication_pad1d_backward(const at::Tensor& grad_output,
                                               const at::Tensor& self,
                                               at::IntArrayRef padding);

  static at::Tensor replication_pad2d(const at::Tensor& self,
                                      at::IntArrayRef padding);
  static at::Tensor replication_pad2d_backward(const at::Tensor& grad_output,
                                               const at::Tensor& self,
                                               at::IntArrayRef padding);

  static at::Tensor& resize_(at::Tensor& self, at::IntArrayRef size,
                             c10::optional<at::MemoryFormat> memory_format);

  static at::Tensor round(const at::Tensor& self);

  static at::Tensor& round_(at::Tensor& self);

  static at::Tensor rrelu_with_noise(const at::Tensor& self,
                                     const at::Tensor& noise, at::Scalar lower,
                                     at::Scalar upper, bool training,
                                     c10::optional<at::Generator> generator);

  static at::Tensor rrelu_with_noise_backward(const at::Tensor& grad_output,
                                              const at::Tensor& self,
                                              const at::Tensor& noise,
                                              at::Scalar lower,
                                              at::Scalar upper, bool training,
                                              bool self_is_result);

  static at::Tensor rsqrt(const at::Tensor& self);

  static at::Tensor& rsqrt_(at::Tensor& self);

  static at::Tensor rsub(const at::Tensor& self, const at::Tensor& other,
                         at::Scalar alpha);

  static at::Tensor rsub(const at::Tensor& self, at::Scalar other,
                         at::Scalar alpha);

  static at::Tensor& scatter_(at::Tensor& self, int64_t dim,
                              const at::Tensor& index, const at::Tensor& src);

  static at::Tensor& scatter_(at::Tensor& self, int64_t dim,
                              const at::Tensor& index, at::Scalar value);

  static at::Tensor& scatter_add_(at::Tensor& self, int64_t dim,
                                  const at::Tensor& index,
                                  const at::Tensor& src);

  static at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index);

  static at::Tensor sigmoid(const at::Tensor& self);

  static at::Tensor& sigmoid_(at::Tensor& self);

  static at::Tensor sigmoid_backward(const at::Tensor& grad_output,
                                     const at::Tensor& output);

  static at::Tensor sign(const at::Tensor& self);

  static at::Tensor& sign_(at::Tensor& self);

  static at::Tensor sin(const at::Tensor& self);

  static at::Tensor& sin_(at::Tensor& self);

  static at::Tensor sinh(const at::Tensor& self);

  static at::Tensor& sinh_(at::Tensor& self);

  static at::Tensor slice(const at::Tensor& self, int64_t dim, int64_t start,
                          int64_t end, int64_t step);

  static at::Tensor smooth_l1_loss(const at::Tensor& self,
                                   const at::Tensor& target, int64_t reduction,
                                   double beta);

  static at::Tensor smooth_l1_loss_backward(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            const at::Tensor& target,
                                            int64_t reduction, double beta);

  static at::Tensor softplus(const at::Tensor& self, at::Scalar beta,
                             at::Scalar threshold);

  static at::Tensor softplus_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self, at::Scalar beta,
                                      at::Scalar threshold,
                                      const at::Tensor& output);

  static at::Tensor softshrink(const at::Tensor& self, at::Scalar lambda);

  static at::Tensor softshrink_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        at::Scalar lambda);

  static std::tuple<at::Tensor, at::Tensor> sort(const at::Tensor& self,
                                                 int64_t dim, bool descending);

  static std::vector<at::Tensor> split(const at::Tensor& self,
                                       int64_t split_size, int64_t dim);

  static std::vector<at::Tensor> split_with_sizes(const at::Tensor& self,
                                                  at::IntArrayRef split_sizes,
                                                  int64_t dim);

  static at::Tensor sqrt(const at::Tensor& self);

  static at::Tensor& sqrt_(at::Tensor& self);

  static at::Tensor squeeze(const at::Tensor& self);

  static at::Tensor squeeze(const at::Tensor& self, int64_t dim);

  static at::Tensor& squeeze_(at::Tensor& self);

  static at::Tensor& squeeze_(at::Tensor& self, int64_t dim);

  static at::Tensor stack(at::TensorList tensors, int64_t dim);

  static at::Tensor std(const at::Tensor& self, bool unbiased);

  static at::Tensor std(const at::Tensor& self, at::IntArrayRef dim,
                        bool unbiased, bool keepdim);

  static at::Tensor sub(const at::Tensor& self, const at::Tensor& other,
                        at::Scalar alpha);

  static at::Tensor sub(const at::Tensor& self, at::Scalar other,
                        at::Scalar alpha);

  static at::Tensor& sub_(at::Tensor& self, const at::Tensor& other,
                          at::Scalar alpha);

  static at::Tensor& sub_(at::Tensor& self, at::Scalar other, at::Scalar alpha);

  static at::Tensor sum(const at::Tensor& self,
                        c10::optional<at::ScalarType> dtype);

  static at::Tensor sum(const at::Tensor& self, at::IntArrayRef dim,
                        bool keepdim, c10::optional<at::ScalarType> dtype);

  static std::tuple<at::Tensor, at::Tensor, at::Tensor> svd(
      const at::Tensor& self, bool some, bool compute_uv);

  static std::tuple<at::Tensor, at::Tensor> symeig(const at::Tensor& self,
                                                   bool eigenvectors,
                                                   bool upper);

  static at::Tensor t(const at::Tensor& self);

  static at::Tensor& t_(at::Tensor& self);

  static at::Tensor take(const at::Tensor& self, const at::Tensor& index);

  static at::Tensor tan(const at::Tensor& self);

  static at::Tensor& tan_(at::Tensor& self);

  static at::Tensor tanh(const at::Tensor& self);

  static at::Tensor& tanh_(at::Tensor& self);

  static at::Tensor tanh_backward(const at::Tensor& grad_output,
                                  const at::Tensor& output);

  static at::Tensor threshold(const at::Tensor& self, at::Scalar threshold,
                              at::Scalar value);

  static at::Tensor& threshold_(at::Tensor& self, at::Scalar threshold,
                                at::Scalar value);

  static at::Tensor threshold_backward(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       at::Scalar threshold);

  static std::tuple<at::Tensor, at::Tensor> topk(const at::Tensor& self,
                                                 int64_t k, int64_t dim,
                                                 bool largest, bool sorted);

  static at::Tensor trace(const at::Tensor& self);

  static std::tuple<at::Tensor, at::Tensor> triangular_solve(
      const at::Tensor& b, const at::Tensor& A, bool upper, bool transpose,
      bool unitriangular);

  static at::Tensor transpose(const at::Tensor& self, int64_t dim0,
                              int64_t dim1);

  static at::Tensor& transpose_(at::Tensor& self, int64_t dim0, int64_t dim1);

  static at::Tensor tril(const at::Tensor& self, int64_t diagonal);

  static at::Tensor& tril_(at::Tensor& self, int64_t diagonal);

  static at::Tensor triu(const at::Tensor& self, int64_t diagonal);

  static at::Tensor& triu_(at::Tensor& self, int64_t diagonal);

  static at::Tensor trunc(const at::Tensor& self);

  static at::Tensor& trunc_(at::Tensor& self);

  static std::vector<at::Tensor> unbind(const at::Tensor& self, int64_t dim);

  static at::Tensor& uniform_(at::Tensor& self, double from, double to,
                              c10::optional<at::Generator> generator);

  static at::Tensor unsqueeze(const at::Tensor& self, int64_t dim);

  static at::Tensor& unsqueeze_(at::Tensor& self, int64_t dim);

  static at::Tensor upsample_bilinear2d(const at::Tensor& self,
                                        at::IntArrayRef output_size,
                                        bool align_corners,
                                        c10::optional<double> scales_h,
                                        c10::optional<double> scales_w);

  static at::Tensor upsample_bilinear2d_backward(
      const at::Tensor& grad_output, at::IntArrayRef output_size,
      at::IntArrayRef input_size, bool align_corners,
      c10::optional<double> scales_h, c10::optional<double> scales_w);

  static at::Tensor upsample_nearest2d(
      const at::Tensor& input, c10::optional<at::IntArrayRef> output_size,
      c10::optional<at::ArrayRef<double>> scale_factors);

  static at::Tensor upsample_nearest2d_backward(
      const at::Tensor& grad_output, c10::optional<at::IntArrayRef> output_size,
      at::IntArrayRef input_size,
      c10::optional<at::ArrayRef<double>> scale_factors);

  static at::Tensor upsample_nearest2d(const at::Tensor& self,
                                       at::IntArrayRef output_size,
                                       c10::optional<double> scales_h,
                                       c10::optional<double> scales_w);

  static at::Tensor upsample_nearest2d_backward(const at::Tensor& grad_output,
                                                at::IntArrayRef output_size,
                                                at::IntArrayRef input_size,
                                                c10::optional<double> scales_h,
                                                c10::optional<double> scales_w);

  static at::Tensor var(const at::Tensor& self, bool unbiased);

  static at::Tensor var(const at::Tensor& self, at::IntArrayRef dim,
                        bool unbiased, bool keepdim);

  static at::Tensor view(const at::Tensor& self, at::IntArrayRef size);

  static at::Tensor& zero_(at::Tensor& self);
};

}  // namespace torch_xla
