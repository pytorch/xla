#include "tensor_impl.h"

#include "tensor_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {

// TODO: Replace UndefinedTensorId with proper type.
XLATensorImpl::XLATensorImpl(XLATensor tensor)
    : c10::TensorImpl(c10::UndefinedTensorId(), GetTypeMeta(tensor),
                      /*allocator=*/nullptr, /*is_variable=*/false),
      tensor_(std::move(tensor)) {
  // Fill up the basic dimension data members which the base class
  // implementation uses in its APIs.
  auto shape = tensor_.shape();
  for (auto dim : shape.get().dimensions()) {
    sizes_.push_back(dim);
    numel_ *= dim;
  }
  for (auto stride : ComputeShapeStrides(shape.get())) {
    strides_.push_back(stride);
  }
}

caffe2::TypeMeta XLATensorImpl::GetTypeMeta(const XLATensor& tensor) {
  auto shape = tensor.shape();
  switch (shape.get().element_type()) {
    case xla::PrimitiveType::F32:
      return caffe2::TypeMeta::Make<float>();
    case xla::PrimitiveType::U8:
      return caffe2::TypeMeta::Make<uint8_t>();
    case xla::PrimitiveType::S8:
      return caffe2::TypeMeta::Make<int8_t>();
    case xla::PrimitiveType::S16:
      return caffe2::TypeMeta::Make<int16_t>();
    case xla::PrimitiveType::S32:
      return caffe2::TypeMeta::Make<int32_t>();
    case xla::PrimitiveType::S64:
      return caffe2::TypeMeta::Make<int64_t>();
    default:
      XLA_ERROR() << "Type not supported: " << shape;
  }
}

}  // namespace torch_xla
