#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

class TokenHandler {
 public:
  explicit TokenHandler(xla::XlaOp token) : token_(token) {}

  xla::XlaOp GetInput(xla::XlaOp input, const xla::Shape* input_shape);

  xla::XlaOp GetNewToken(xla::XlaOp result);

 private:
  xla::XlaOp token_;
};

}  // namespace torch_xla
