#pragma once

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

struct ConvOpAttrs {
  bool depthwise;
  int num_spatial_dims;
  std::vector<xla::int32> dilations;
  std::vector<xla::int32> strides;
  tensorflow::Padding padding;
  std::vector<xla::int64> explicit_paddings;
  tensorflow::TensorFormat data_format;
};

inline xla::StatusOr<xla::XlaOp> MakeXlaBackpropInputConvOp(
    std::string type_string, const xla::Shape& input_shape, xla::XlaOp filter,
    xla::XlaOp out_backprop, const ConvOpAttrs& attrs,
    const xla::PrecisionConfig* precision_config = nullptr) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline xla::StatusOr<xla::XlaOp> MakeXlaBackpropFilterConvOp(
    std::string type_string, xla::XlaOp activations,
    const xla::Shape& filter_shape, xla::XlaOp gradients,
    const ConvOpAttrs& attrs,
    const xla::PrecisionConfig* precision_config = nullptr) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace tensorflow
