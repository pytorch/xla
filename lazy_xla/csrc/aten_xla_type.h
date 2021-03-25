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
  static at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst,
                               bool non_blocking);

  static at::Tensor _s_where(const at::Tensor& condition,
                             const at::Tensor& self, const at::Tensor& other);

  static at::Tensor _trilinear(const at::Tensor& i1, const at::Tensor& i2,
                               const at::Tensor& i3, at::IntArrayRef expand1,
                               at::IntArrayRef expand2, at::IntArrayRef expand3,
                               at::IntArrayRef sumdim, int64_t unroll_dim);

  static at::Tensor _unsafe_view(const at::Tensor& self, at::IntArrayRef size);

  static at::Tensor abs(const at::Tensor& self);

  static at::Tensor& abs_(at::Tensor& self);

  static at::Tensor acos(const at::Tensor& self);

  static at::Tensor acosh(const at::Tensor& self);

  static at::Tensor& acos_(at::Tensor& self);

  static at::Tensor add(const at::Tensor& self, const at::Tensor& other,
                        const at::Scalar& alpha);

  static at::Tensor add(const at::Tensor& self, const at::Scalar& other,
                        const at::Scalar& alpha);

  static at::Tensor& add_(at::Tensor& self, const at::Tensor& other,
                          const at::Scalar& alpha);

  static at::Tensor& add_(at::Tensor& self, const at::Scalar& other,
                          const at::Scalar& alpha);

  static at::Tensor addcdiv(const at::Tensor& self, const at::Tensor& tensor1,
                            const at::Tensor& tensor2, const at::Scalar& value);

  static at::Tensor& addcdiv_(at::Tensor& self, const at::Tensor& tensor1,
                              const at::Tensor& tensor2,
                              const at::Scalar& value);

  static at::Tensor addcmul(const at::Tensor& self, const at::Tensor& tensor1,
                            const at::Tensor& tensor2, const at::Scalar& value);

  static at::Tensor& addcmul_(at::Tensor& self, const at::Tensor& tensor1,
                              const at::Tensor& tensor2,
                              const at::Scalar& value);

  static at::Tensor alias(const at::Tensor& self);

  static at::Tensor as_strided(const at::Tensor& self, at::IntArrayRef size,
                               at::IntArrayRef stride,
                               c10::optional<int64_t> storage_offset);

  static at::Tensor& as_strided_(at::Tensor& self, at::IntArrayRef size,
                                 at::IntArrayRef stride,
                                 c10::optional<int64_t> storage_offset);

  static at::Tensor asin(const at::Tensor& self);

  static at::Tensor& asin_(at::Tensor& self);

  static at::Tensor asinh(const at::Tensor& self);

  static at::Tensor atan(const at::Tensor& self);

  static at::Tensor atanh(const at::Tensor& self);

  static at::Tensor atan2(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& atan2_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& atan_(at::Tensor& self);

  static at::Tensor& bitwise_and_out(const at::Tensor& self,
                                     const at::Tensor& other, at::Tensor& out);

  static at::Tensor& bitwise_and_out(const at::Tensor& self,
                                     const at::Scalar& other, at::Tensor& out);

  static at::Tensor& bitwise_or_out(const at::Tensor& self,
                                    const at::Tensor& other, at::Tensor& out);

  static at::Tensor& bitwise_or_out(const at::Tensor& self,
                                    const at::Scalar& other, at::Tensor& out);

  static at::Tensor& bitwise_xor_out(const at::Tensor& self,
                                     const at::Scalar& other, at::Tensor& out);

  static at::Tensor& bitwise_xor_out(const at::Tensor& self,
                                     const at::Tensor& other, at::Tensor& out);

  static at::Tensor ceil(const at::Tensor& self);

  static at::Tensor& ceil_(at::Tensor& self);

  static at::Tensor clamp(const at::Tensor& self,
                          const c10::optional<at::Scalar>& min,
                          const c10::optional<at::Scalar>& max);

  static at::Tensor& clamp_(at::Tensor& self,
                            const c10::optional<at::Scalar>& min,
                            const c10::optional<at::Scalar>& max);

  static at::Tensor clamp_max(const at::Tensor& self, const at::Scalar& max);

  static at::Tensor& clamp_max_(at::Tensor& self, const at::Scalar& max);

  static at::Tensor clamp_min(const at::Tensor& self, const at::Scalar& min);

  static at::Tensor& clamp_min_(at::Tensor& self, const at::Scalar& min);

  static at::Tensor clone(const at::Tensor& self,
                          c10::optional<at::MemoryFormat> memory_format);

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

  static at::Tensor diagonal(const at::Tensor& self, int64_t offset,
                             int64_t dim1, int64_t dim2);

  static at::Tensor div(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor div(const at::Tensor& self, const at::Tensor& other,
                        std::string rounding_mode);

  static at::Tensor div(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor& div_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& div_(at::Tensor& self, const at::Tensor& other,
                          std::string rounding_mode);

  static at::Tensor& div_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor elu(const at::Tensor& self, const at::Scalar& alpha,
                        const at::Scalar& scale, const at::Scalar& input_scale);

  static at::Tensor& elu_(at::Tensor& self, const at::Scalar& alpha,
                          const at::Scalar& scale,
                          const at::Scalar& input_scale);

  static at::Tensor elu_backward(const at::Tensor& grad_output,
                                 const at::Scalar& alpha,
                                 const at::Scalar& scale,
                                 const at::Scalar& input_scale, bool self,
                                 const at::Tensor& self_or_result);

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

  static at::Tensor eq(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor eq(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& eq_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor& eq_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor erf(const at::Tensor& self);

  static at::Tensor& erf_(at::Tensor& self);

  static at::Tensor erfc(const at::Tensor& self);

  static at::Tensor& erfc_(at::Tensor& self);

  static at::Tensor exp(const at::Tensor& self);

  static at::Tensor& exp_(at::Tensor& self);

  static at::Tensor expand(const at::Tensor& self, at::IntArrayRef size,
                           bool implicit);

  static at::Tensor expm1(const at::Tensor& self);

  static at::Tensor& expm1_(at::Tensor& self);

  static at::Tensor& fill_(at::Tensor& self, const at::Scalar& value);

  static at::Tensor& fill_(at::Tensor& self, const at::Tensor& value);

  static at::Tensor floor(const at::Tensor& self);

  static at::Tensor& floor_(at::Tensor& self);

  static at::Tensor fmod(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor fmod(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor& fmod_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& fmod_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor frac(const at::Tensor& self);

  static at::Tensor& frac_(at::Tensor& self);

  static at::Tensor ge(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor ge(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& ge_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor& ge_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor gelu(const at::Tensor& self);

  static at::Tensor gelu_backward(const at::Tensor& grad,
                                  const at::Tensor& self);

  static at::Tensor gt(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor gt(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& gt_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor& gt_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor hardtanh(const at::Tensor& self, const at::Scalar& min_val,
                             const at::Scalar& max_val);

  static at::Tensor& hardtanh_(at::Tensor& self, const at::Scalar& min_val,
                               const at::Scalar& max_val);

  static at::Tensor hardtanh_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      const at::Scalar& min_val,
                                      const at::Scalar& max_val);

  static at::Tensor kl_div(const at::Tensor& self, const at::Tensor& target,
                           int64_t reduction, bool log_target);

  static at::Tensor kl_div_backward(const at::Tensor& grad_output,
                                    const at::Tensor& self,
                                    const at::Tensor& target, int64_t reduction,
                                    bool log_target);

  static at::Tensor le(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor le(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& le_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor& le_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor leaky_relu(const at::Tensor& self,
                               const at::Scalar& negative_slope);

  static at::Tensor& leaky_relu_(at::Tensor& self,
                                 const at::Scalar& negative_slope);

  static at::Tensor leaky_relu_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Scalar& negative_slope,
                                        bool self_is_result);

  static at::Tensor log(const at::Tensor& self);

  static at::Tensor log10(const at::Tensor& self);

  static at::Tensor log1p(const at::Tensor& self);

  static at::Tensor& log1p_(at::Tensor& self);

  static at::Tensor log2(const at::Tensor& self);

  static at::Tensor& log_(at::Tensor& self);

  static at::Tensor log_sigmoid_backward(const at::Tensor& grad_output,
                                         const at::Tensor& self,
                                         const at::Tensor& buffer);

  static std::tuple<at::Tensor, at::Tensor> log_sigmoid_forward(
      const at::Tensor& self);

  static at::Tensor logsumexp(const at::Tensor& self, at::IntArrayRef dim,
                              bool keepdim);

  static at::Tensor lt(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor lt(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& lt_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor& lt_(at::Tensor& self, const at::Tensor& other);

  static std::tuple<at::Tensor, at::Tensor> max(const at::Tensor& self,
                                                int64_t dim, bool keepdim);

  static at::Tensor maximum(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor max_pool2d(const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool ceil_mode);

  static at::Tensor max_pool3d(const at::Tensor& self,
                               at::IntArrayRef kernel_size,
                               at::IntArrayRef stride, at::IntArrayRef padding,
                               at::IntArrayRef dilation, bool ceil_mode);

  static std::tuple<at::Tensor, at::Tensor> min(const at::Tensor& self,
                                                int64_t dim, bool keepdim);

  static at::Tensor minimum(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor mul(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor mul(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor& mul_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& mul_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor ne(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor ne(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor& ne_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor& ne_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor neg(const at::Tensor& self);

  static at::Tensor& neg_(at::Tensor& self);

  static at::Tensor norm(const at::Tensor& self,
                         const c10::optional<at::Scalar>& p,
                         at::ScalarType dtype);

  static at::Tensor norm(const at::Tensor& self, const at::Scalar& p);

  static at::Tensor norm(const at::Tensor& self,
                         const c10::optional<at::Scalar>& p,
                         at::IntArrayRef dim, bool keepdim,
                         at::ScalarType dtype);

  static at::Tensor norm(const at::Tensor& self,
                         const c10::optional<at::Scalar>& p,
                         at::IntArrayRef dim, bool keepdim);

  static at::Tensor permute(const at::Tensor& self, at::IntArrayRef dims);

  static at::Tensor pow(const at::Tensor& self, const at::Scalar& exponent);

  static at::Tensor pow(const at::Tensor& self, const at::Tensor& exponent);

  static at::Tensor pow(const at::Scalar& self, const at::Tensor& exponent);

  static at::Tensor& pow_(at::Tensor& self, const at::Scalar& exponent);

  static at::Tensor& pow_(at::Tensor& self, const at::Tensor& exponent);

  static at::Tensor reciprocal(const at::Tensor& self);

  static at::Tensor& reciprocal_(at::Tensor& self);

  static at::Tensor relu(const at::Tensor& self);

  static at::Tensor& relu_(at::Tensor& self);

  static at::Tensor remainder(const at::Tensor& self, const at::Tensor& other);

  static at::Tensor remainder(const at::Tensor& self, const at::Scalar& other);

  static at::Tensor& remainder_(at::Tensor& self, const at::Tensor& other);

  static at::Tensor& remainder_(at::Tensor& self, const at::Scalar& other);

  static at::Tensor repeat(const at::Tensor& self, at::IntArrayRef repeats);

  static at::Tensor& resize_(at::Tensor& self, at::IntArrayRef size,
                             c10::optional<at::MemoryFormat> memory_format);

  static at::Tensor round(const at::Tensor& self);

  static at::Tensor& round_(at::Tensor& self);

  static at::Tensor rrelu_with_noise(const at::Tensor& self,
                                     const at::Tensor& noise,
                                     const at::Scalar& lower,
                                     const at::Scalar& upper, bool training,
                                     c10::optional<at::Generator> generator);

  static at::Tensor rrelu_with_noise_backward(
      const at::Tensor& grad_output, const at::Tensor& self,
      const at::Tensor& noise, const at::Scalar& lower, const at::Scalar& upper,
      bool training, bool self_is_result);

  static at::Tensor rsqrt(const at::Tensor& self);

  static at::Tensor& rsqrt_(at::Tensor& self);

  static at::Tensor rsub(const at::Tensor& self, const at::Tensor& other,
                         const at::Scalar& alpha);

  static at::Tensor rsub(const at::Tensor& self, const at::Scalar& other,
                         const at::Scalar& alpha);

  static at::Tensor select(const at::Tensor& self, int64_t dim, int64_t index);

  static at::Tensor& silu_out(const at::Tensor& self, at::Tensor& out);

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

  static at::Tensor slice(const at::Tensor& self, int64_t dim,
                          c10::optional<int64_t> start,
                          c10::optional<int64_t> end, int64_t step);

  static at::Tensor softplus(const at::Tensor& self, const at::Scalar& beta,
                             const at::Scalar& threshold);

  static at::Tensor softplus_backward(const at::Tensor& grad_output,
                                      const at::Tensor& self,
                                      const at::Scalar& beta,
                                      const at::Scalar& threshold,
                                      const at::Tensor& output);

  static at::Tensor softshrink(const at::Tensor& self,
                               const at::Scalar& lambda);

  static at::Tensor softshrink_backward(const at::Tensor& grad_output,
                                        const at::Tensor& self,
                                        const at::Scalar& lambda);

  static at::Tensor sqrt(const at::Tensor& self);

  static at::Tensor& sqrt_(at::Tensor& self);

  static at::Tensor& squeeze_(at::Tensor& self);

  static at::Tensor& squeeze_(at::Tensor& self, int64_t dim);

  static at::Tensor sub(const at::Tensor& self, const at::Tensor& other,
                        const at::Scalar& alpha);

  static at::Tensor sub(const at::Tensor& self, const at::Scalar& other,
                        const at::Scalar& alpha);

  static at::Tensor& sub_(at::Tensor& self, const at::Tensor& other,
                          const at::Scalar& alpha);

  static at::Tensor& sub_(at::Tensor& self, const at::Scalar& other,
                          const at::Scalar& alpha);

  static at::Tensor t(const at::Tensor& self);

  static at::Tensor& t_(at::Tensor& self);

  static at::Tensor tan(const at::Tensor& self);

  static at::Tensor& tan_(at::Tensor& self);

  static at::Tensor tanh(const at::Tensor& self);

  static at::Tensor& tanh_(at::Tensor& self);

  static at::Tensor tanh_backward(const at::Tensor& grad_output,
                                  const at::Tensor& output);

  static at::Tensor threshold(const at::Tensor& self,
                              const at::Scalar& threshold,
                              const at::Scalar& value);

  static at::Tensor& threshold_(at::Tensor& self, const at::Scalar& threshold,
                                const at::Scalar& value);

  static at::Tensor threshold_backward(const at::Tensor& grad_output,
                                       const at::Tensor& self,
                                       const at::Scalar& threshold);

  static at::Tensor transpose(const at::Tensor& self, int64_t dim0,
                              int64_t dim1);

  static at::Tensor& transpose_(at::Tensor& self, int64_t dim0, int64_t dim1);

  static at::Tensor trunc(const at::Tensor& self);

  static at::Tensor& trunc_(at::Tensor& self);

  static at::Tensor view(const at::Tensor& self, at::IntArrayRef size);

  static at::Tensor& zero_(at::Tensor& self);
};

}  // namespace torch_xla
