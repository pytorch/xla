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

  std::vector<int64_t> byte_strides() const {
    std::vector<int64_t> byte_strides(shape().dimensions_size());
    XLA_CHECK_OK(
        xla::ShapeUtil::ByteStrides(shape(), absl::MakeSpan(byte_strides)));
    return byte_strides;
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
