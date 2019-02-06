#include "aten_xla_type.h"

#include <mutex>

#include "aten_xla_bridge.h"
#include "aten_xla_type_instances.h"
#include "helpers.h"
#include "tensor_impl.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace {

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

void AtenXlaType::SetFullConvPrecision(
    bool use_full_conv_precision /*= true*/) {
  s_use_full_conv_precision_ = use_full_conv_precision;
}

void AtenXlaType::RegisterAtenTypes() {
  static std::once_flag once;
  std::call_once(once, []() { RegisterAtenXlaTypes(); });
}

}  // namespace torch_xla
