#ifndef XLA_CLIENT_TENSOR_SOURCE_H_
#define XLA_CLIENT_TENSOR_SOURCE_H_

#include <ATen/Tensor.h>
#include <torch/csrc/lazy/core/metrics.h>

#include <vector>

#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace runtime {

// Owns a contiguous block of data with the shape and layout matching `shape()`.
class TensorSource {
 public:
  TensorSource(std::string device) : device_(std::move(device)){};

  virtual const void* data() const = 0;

  virtual xla::PrimitiveType primitive_type() const = 0;

  virtual std::vector<int64_t> dimensions() const = 0;

  virtual std::vector<int64_t> byte_strides() const = 0;

  const std::string& device() const { return device_; }

 private:
  std::string device_;
};

class AtenSource : public TensorSource {
 public:
  AtenSource(const at::Tensor& tensor, xla::PrimitiveType target_type,
             std::string device)
      : TensorSource(std::move(device)), target_type_(target_type) {
    at::ScalarType target_torch_type = TorchTypeFromXlaType(primitive_type());
    if (target_torch_type != tensor.type().scalarType()) {
      TORCH_LAZY_COUNTER("AtenSourceDowncasts", 1);
      tensor_ = std::move(tensor.to(target_torch_type).contiguous());
    } else {
      tensor_ = std::move(tensor.contiguous());
    }
  }

  const void* data() const override { return tensor_.const_data_ptr(); }

  xla::PrimitiveType primitive_type() const override { return target_type_; }

  std::vector<int64_t> dimensions() const override {
    auto sizes = tensor_.sizes();
    return {sizes.begin(), sizes.end()};
  }

  std::vector<int64_t> byte_strides() const override {
    std::vector<int64_t> strides;
    for (auto& stride : tensor_.strides()) {
      strides.push_back(stride * tensor_.itemsize());
    }
    return strides;
  }

 private:
  at::Tensor tensor_;
  xla::PrimitiveType target_type_;
};

class LiteralSource : public TensorSource {
 public:
  LiteralSource(xla::Literal literal, std::string device)
      : TensorSource(std::move(device)), literal_(std::move(literal)) {}

  const void* data() const override { return literal_.untyped_data(); }

  const xla::Shape& shape() const { return literal_.shape(); }

  xla::PrimitiveType primitive_type() const override {
    return shape().element_type();
  }

  std::vector<int64_t> dimensions() const override {
    auto dimensions = shape().dimensions();
    return {dimensions.begin(), dimensions.end()};
  }

  std::vector<int64_t> byte_strides() const override {
    std::vector<int64_t> byte_strides(shape().dimensions_size());
    XLA_CHECK_OK(
        xla::ShapeUtil::ByteStrides(shape(), absl::MakeSpan(byte_strides)));
    return byte_strides;
  }

 private:
  xla::Literal literal_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_COMPUTATION_CLIENT_H_
