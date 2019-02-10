#include "aten_xla_type.h"

#include <mutex>

#include "aten_xla_bridge.h"
#include "aten_xla_type_instances.h"
#include "device.h"
#include "helpers.h"
#include "tensor_impl.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace {

struct XlaOptions {
  XlaOptions(const at::TensorOptions& options,
             c10::optional<Device> device = c10::nullopt)
      : device(std::move(device)) {
    if (options.has_device()) {
      device = bridge::AtenDeviceToXlaDevice(options.device());
    }
    if (options.has_dtype()) {
      scalar_type = c10::typeMetaToScalarType(options.dtype());
    }
  }

  Device get_device() const { return device ? *device : *GetDefaultDevice(); }

  at::ScalarType get_scalar_type() const {
    return scalar_type ? *scalar_type : at::ScalarType::Float;
  }

  c10::optional<Device> device;
  c10::optional<at::ScalarType> scalar_type;
};

// Returns true if dilation is non-trivial (not 1) in at least one dimension.
bool IsNonTrivialDilation(at::IntList dilation) {
  return std::any_of(
      dilation.begin(), dilation.end(),
      [](const int64_t dim_dilation) { return dim_dilation != 1; });
}

}  // namespace

bool AtenXlaType::s_use_full_conv_precision_ = false;

AtenXlaType::AtenXlaType(at::TensorTypeId type_id, bool is_variable,
                         bool is_undefined)
    : AtenXlaTypeBase(type_id, is_variable, is_undefined) {}

at::Tensor AtenXlaType::zeros(at::IntArrayRef size,
                              const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(
      XLATensor::zeros(XlaHelpers::I64List(size), xla_options.get_device(),
                       xla_options.get_scalar_type()));
}

at::Tensor AtenXlaType::zeros_like(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(XLATensor::zeros_like(
      self_tensor, self_tensor.GetDevice(), c10::nullopt));
}

at::Tensor AtenXlaType::zeros_like(const at::Tensor& self,
                                   const at::TensorOptions& options) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XlaOptions xla_options(options, self_tensor.GetDevice());
  return bridge::AtenFromXlaTensor(XLATensor::zeros_like(
      self_tensor, xla_options.get_device(), xla_options.scalar_type));
}

at::Tensor AtenXlaType::ones(at::IntArrayRef size,
                             const at::TensorOptions& options) const {
  XlaOptions xla_options(options);
  return bridge::AtenFromXlaTensor(
      XLATensor::ones(XlaHelpers::I64List(size), xla_options.get_device(),
                      xla_options.get_scalar_type()));
}

at::Tensor AtenXlaType::ones_like(const at::Tensor& self) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  return bridge::AtenFromXlaTensor(
      XLATensor::ones_like(self_tensor, self_tensor.GetDevice(), c10::nullopt));
}

at::Tensor AtenXlaType::ones_like(const at::Tensor& self,
                                  const at::TensorOptions& options) const {
  XLATensor self_tensor = bridge::GetXlaTensor(self);
  XlaOptions xla_options(options, self_tensor.GetDevice());
  return bridge::AtenFromXlaTensor(XLATensor::ones_like(
      self_tensor, xla_options.get_device(), xla_options.scalar_type));
}

at::Tensor AtenXlaType::add(const at::Tensor& self, const at::Tensor& other,
                            at::Scalar alpha) const {
  return bridge::AtenFromXlaTensor(
      bridge::GetXlaTensor(self).add(bridge::GetXlaTensor(other), alpha));
}

at::Tensor& AtenXlaType::add_(at::Tensor& self, const at::Tensor& other,
                              at::Scalar alpha) const {
  bridge::GetXlaTensor(self).add_(bridge::GetXlaTensor(other), alpha);
  return self;
}

at::Tensor AtenXlaType::mul(const at::Tensor& self,
                            const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      bridge::GetXlaTensor(self).mul(bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::mul_(at::Tensor& self, const at::Tensor& other) const {
  bridge::GetXlaTensor(self).mul_(bridge::GetXlaTensor(other));
  return self;
}

at::Tensor AtenXlaType::div(const at::Tensor& self,
                            const at::Tensor& other) const {
  return bridge::AtenFromXlaTensor(
      bridge::GetXlaTensor(self).div(bridge::GetXlaTensor(other)));
}

at::Tensor& AtenXlaType::div_(at::Tensor& self, const at::Tensor& other) const {
  bridge::GetXlaTensor(self).div_(bridge::GetXlaTensor(other));
  return self;
}

int64_t AtenXlaType::size(const at::Tensor& self, int64_t dim) const {
  return bridge::GetXlaTensor(self).size(dim);
}

at::Tensor AtenXlaType::relu(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(self).relu());
}

at::Tensor AtenXlaType::threshold(const at::Tensor& self, at::Scalar threshold,
                                  at::Scalar value) const {
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(self).threshold(
      threshold.to<double>(), value.to<double>()));
}

at::Tensor AtenXlaType::threshold_backward(const at::Tensor& grad_output,
                                           const at::Tensor& self,
                                           at::Scalar threshold) const {
  return bridge::AtenFromXlaTensor(XLATensor::threshold_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      threshold.to<double>()));
}

at::Tensor AtenXlaType::conv2d(const at::Tensor& input,
                               const at::Tensor& weight, const at::Tensor& bias,
                               at::IntList stride, at::IntList padding,
                               at::IntList dilation, int64_t groups) const {
  // Dilated or grouped convolutions aren't lowered to XLA yet.
  if (IsNonTrivialDilation(dilation) || groups != 1) {
    return AtenXlaTypeBase::conv2d(input, weight, bias, stride, padding,
                                   dilation, groups);
  }
  if (bias.defined()) {
    return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(input).conv2d(
        bridge::GetXlaTensor(weight), bridge::GetXlaTensor(bias),
        XlaHelpers::I64List(stride), XlaHelpers::I64List(padding),
        /*use_full_conv_precision=*/s_use_full_conv_precision_));
  } else {
    return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(input).conv2d(
        bridge::GetXlaTensor(weight), XlaHelpers::I64List(stride),
        XlaHelpers::I64List(padding),
        /*use_full_conv_precision=*/s_use_full_conv_precision_));
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> AtenXlaType::thnn_conv2d_forward(
    const at::Tensor& self, const at::Tensor& weight,
    at::IntArrayRef kernel_size, const at::Tensor& bias, at::IntArrayRef stride,
    at::IntArrayRef padding) const {
  at::Tensor undefined = at::empty({});
  // TODO(asuhan): double check it's ok to return undefined for finput and
  // fgrad_input.
  return std::make_tuple(
      conv2d(/*input=*/self, /*weight=*/weight, /*bias=*/bias,
             /*stride=*/stride, /*padding=*/padding, /*dilation=*/{1, 1},
             /*groups=*/1),
      undefined, undefined);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
AtenXlaType::thnn_conv2d_backward(const at::Tensor& grad_output,
                                  const at::Tensor& self,
                                  const at::Tensor& weight,
                                  at::IntList kernel_size, at::IntList stride,
                                  at::IntList padding, const at::Tensor& finput,
                                  const at::Tensor& fgrad_input,
                                  std::array<bool, 3> output_mask) const {
  at::Tensor undefined;
  auto gradients = XLATensor::conv2d_backward(
      /*out_backprop=*/bridge::GetXlaTensor(grad_output),
      /*input=*/bridge::GetXlaTensor(self),
      /*weight=*/bridge::GetXlaTensor(weight),
      /*stride=*/XlaHelpers::I64List(stride),
      /*padding=*/XlaHelpers::I64List(padding),
      /*use_full_conv_precision=*/s_use_full_conv_precision_);
  return std::make_tuple(
      output_mask[0] ? bridge::AtenFromXlaTensor(std::get<0>(gradients))
                     : undefined,
      output_mask[1] ? bridge::AtenFromXlaTensor(std::get<1>(gradients))
                     : undefined,
      output_mask[2] ? bridge::AtenFromXlaTensor(std::get<2>(gradients))
                     : undefined);
}

at::Tensor AtenXlaType::addmm(const at::Tensor& self, const at::Tensor& mat1,
                              const at::Tensor& mat2, at::Scalar beta,
                              at::Scalar alpha) const {
  if (beta.to<double>() != 1 || alpha.to<double>() != 1) {
    return AtenXlaTypeBase::addmm(self, mat1, mat2, beta, alpha);
  }
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(mat1).addmm(
      /*weight=*/bridge::GetXlaTensor(mat2),
      /*bias=*/bridge::GetXlaTensor(self),
      /*use_full_conv_precision=*/s_use_full_conv_precision_));
}

at::Tensor AtenXlaType::mm(const at::Tensor& self,
                           const at::Tensor& mat2) const {
  return bridge::AtenFromXlaTensor(
      XLATensor::mm(/*input=*/bridge::GetXlaTensor(self),
                    /*weight=*/bridge::GetXlaTensor(mat2),
                    /*use_full_conv_precision=*/s_use_full_conv_precision_));
}

at::Tensor AtenXlaType::t(const at::Tensor& self) const {
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(self).t());
}

at::Tensor AtenXlaType::view(const at::Tensor& self, at::IntList size) const {
  return bridge::AtenFromXlaTensor(
      bridge::GetXlaTensor(self).view(XlaHelpers::I64List(size)));
}

at::Tensor AtenXlaType::log_softmax(const at::Tensor& self, int64_t dim) const {
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(self).log_softmax(dim));
}

at::Tensor AtenXlaType::max_pool2d(const at::Tensor& self,
                                   at::IntList kernel_size, at::IntList stride,
                                   at::IntList padding, at::IntList dilation,
                                   bool ceil_mode) const {
  // Lowering when dilation is non-trivial or ceil_mode is set not supported.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool2d(self, kernel_size, stride, padding,
                                       dilation, ceil_mode);
  }
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(self).max_pool2d(
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding)));
}

std::tuple<at::Tensor, at::Tensor> AtenXlaType::max_pool2d_with_indices(
    const at::Tensor& self, at::IntList kernel_size, at::IntList stride,
    at::IntList padding, at::IntList dilation, bool ceil_mode) const {
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool2d_with_indices(
        self, kernel_size, stride, padding, dilation, ceil_mode);
  }
  // TODO(asuhan): Here we return a placeholder tensor for the indices we hope
  // to never evaluate, which works for the backward of max_pool2d. However, the
  // user could request the indices to be returned, in which case we'd throw. We
  // need to either provide a lowering or improve our infrastructure to be able
  // to route to ATen the evaluation of outputs we hope to be unused.
  XLATensor result = bridge::GetXlaTensor(self).max_pool2d(
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding));
  xla::Shape indices_shape = result.shape();
  indices_shape.set_element_type(xla::PrimitiveType::S64);
  XLATensor indices_not_supported =
      XLATensor::not_supported(at::aten::max_pool2d_with_indices, indices_shape,
                               bridge::GetXlaTensor(self).GetDevice());
  return std::make_tuple(bridge::AtenFromXlaTensor(result),
                         bridge::AtenFromXlaTensor(indices_not_supported));
}

at::Tensor AtenXlaType::avg_pool2d(const at::Tensor& self,
                                   at::IntList kernel_size, at::IntList stride,
                                   at::IntList padding, bool ceil_mode,
                                   bool count_include_pad) const {
  // Lowering when ceil_mode is set not supported yet.
  if (ceil_mode) {
    return AtenXlaTypeBase::avg_pool2d(self, kernel_size, stride, padding,
                                       ceil_mode, count_include_pad);
  }
  return bridge::AtenFromXlaTensor(bridge::GetXlaTensor(self).avg_pool2d(
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), count_include_pad));
}

at::Tensor AtenXlaType::avg_pool2d_backward(const at::Tensor& grad_output,
                                            const at::Tensor& self,
                                            at::IntList kernel_size,
                                            at::IntList stride,
                                            at::IntList padding, bool ceil_mode,
                                            bool count_include_pad) const {
  // Lowering when ceil_mode is set not supported yet.
  if (ceil_mode) {
    return AtenXlaTypeBase::avg_pool2d_backward(grad_output, self, kernel_size,
                                                stride, padding, ceil_mode,
                                                count_include_pad);
  }
  return bridge::AtenFromXlaTensor(XLATensor::avg_pool2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding), count_include_pad));
}

at::Tensor AtenXlaType::max_pool2d_with_indices_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    at::IntList kernel_size, at::IntList stride, at::IntList padding,
    at::IntList dilation, bool ceil_mode, const at::Tensor& indices) const {
  // Lowering when ceil_mode or dilation is set not supported yet.
  if (ceil_mode || IsNonTrivialDilation(dilation)) {
    return AtenXlaTypeBase::max_pool2d_with_indices_backward(
        grad_output, self, kernel_size, stride, padding, dilation, ceil_mode,
        indices);
  }
  return bridge::AtenFromXlaTensor(XLATensor::max_pool2d_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(self),
      XlaHelpers::I64List(kernel_size), XlaHelpers::I64List(stride),
      XlaHelpers::I64List(padding)));
}

at::Tensor AtenXlaType::_log_softmax_backward_data(
    const at::Tensor& grad_output, const at::Tensor& output, int64_t dim,
    const at::Tensor& /* self*/) const {
  return bridge::AtenFromXlaTensor(XLATensor::log_softmax_backward(
      bridge::GetXlaTensor(grad_output), bridge::GetXlaTensor(output), dim));
}

at::Tensor AtenXlaType::nll_loss(const at::Tensor& self,
                                 const at::Tensor& target,
                                 const at::Tensor& weight, int64_t reduction,
                                 int64_t ignore_index) const {
  if (reduction != Reduction::Mean || ignore_index >= 0 || weight.defined()) {
    return AtenXlaTypeBase::nll_loss(self, target, weight, reduction,
                                     ignore_index);
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target)));
}

at::Tensor AtenXlaType::nll_loss_backward(
    const at::Tensor& grad_output, const at::Tensor& self,
    const at::Tensor& target, const at::Tensor& weight, int64_t reduction,
    int64_t ignore_index, const at::Tensor& total_weight) const {
  if (reduction != Reduction::Mean || ignore_index >= 0 || weight.defined()) {
    return AtenXlaTypeBase::nll_loss_backward(grad_output, self, target, weight,
                                              reduction, ignore_index,
                                              total_weight);
  }
  return bridge::AtenFromXlaTensor(XLATensor::nll_loss_backward(
      bridge::GetXlaTensor(self), bridge::GetXlaTensor(target)));
}

void AtenXlaType::SetFullConvPrecision(
    bool use_full_conv_precision /*= true*/) {
  s_use_full_conv_precision_ = use_full_conv_precision;
}

void AtenXlaType::RegisterAtenTypes() {
  static std::once_flag once;
  std::call_once(once, []() { RegisterAtenXlaTypes(); });
}

}  // namespace torch_xla
