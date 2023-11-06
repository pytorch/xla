#ifndef XLA_CLIENT_TENSOR_SOURCE_H_
#define XLA_CLIENT_TENSOR_SOURCE_H_

#include <ATen/Tensor.h>

#include <vector>

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

  virtual const xla::Shape& shape() const = 0;

  const std::string& device() const { return device_; }

  virtual std::vector<int64_t> byte_strides() const {
    std::vector<int64_t> byte_strides(shape().dimensions_size());
    XLA_CHECK_OK(
        xla::ShapeUtil::ByteStrides(shape(), absl::MakeSpan(byte_strides)));
    return byte_strides;
  }

  virtual std::vector<int64_t> dimensions() const {
    auto dimensions = shape().dimensions();
    return {dimensions.begin(), dimensions.end()};
  }

  virtual xla::PrimitiveType primitive_type() const {
    return shape().element_type();
  }

 private:
  std::string device_;
};

class AtenSource : public TensorSource {
 public:
  AtenSource(const at::Tensor& tensor, xla::Shape shape, std::string device)
      : TensorSource(std::move(device)),
        tensor_(std::move(tensor.contiguous())),
        shape_(std::move(shape)) {}

  const void* data() const override { return tensor_.const_data_ptr(); }

  const xla::Shape& shape() const override { return shape_; }

  std::vector<int64_t> byte_strides() const override {
    std::vector<int64_t> strides;
    for (auto& stride : tensor_.strides()) {
      strides.push_back(stride * tensor_.itemsize());
    }
    return strides;
  }

  std::vector<int64_t> dimensions() const override {
    auto sizes = tensor_.sizes();
    return {sizes.begin(), sizes.end()};
  }

  // xla::PrimitiveType primitive_type() const override {
  //   switch (tensor_.type().scalarType()) {
  //     case at::ScalarType::Double:
  //       return xla::PrimitiveType::F64;
  //     case at::ScalarType::Float:
  //       return xla::PrimitiveType::F32;
  //     case at::ScalarType::BFloat16:
  //       return xla::PrimitiveType::BF16;
  //     case at::ScalarType::Half:
  //       return xla::PrimitiveType::F16;
  //     case at::ScalarType::Bool:
  //       return xla::PrimitiveType::PRED;
  //     case at::ScalarType::Byte:
  //       return xla::PrimitiveType::U8;
  //     case at::ScalarType::Char:
  //       return xla::PrimitiveType::S8;
  //     case at::ScalarType::Short:
  //       return xla::PrimitiveType::S16;
  //     case at::ScalarType::Int:
  //       return xla::PrimitiveType::S32;
  //     case at::ScalarType::Long:
  //       return xla::PrimitiveType::S64;
  //     case at::ScalarType::ComplexFloat:
  //       return xla::PrimitiveType::C64;
  //     case at::ScalarType::ComplexDouble:
  //       return xla::PrimitiveType::C128;
  //     default:
  //       XLA_ERROR() << "Type not supported: " << tensor_.type().scalarType();
  //   }
  // }

 private:
  at::Tensor tensor_;
  xla::Shape shape_;
};

class LiteralSource : public TensorSource {
 public:
  LiteralSource(xla::Literal literal, std::string device)
      : TensorSource(std::move(device)), literal_(std::move(literal)) {}

  const void* data() const override { return literal_.untyped_data(); }

  const xla::Shape& shape() const override { return literal_.shape(); }

 private:
  xla::Literal literal_;
};

}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_COMPUTATION_CLIENT_H_
