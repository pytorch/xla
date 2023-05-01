#ifndef XLA_TORCH_XLA_CSRC_NMS_OP_H_
#define XLA_TORCH_XLA_CSRC_NMS_OP_H_

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace torch_xla {

struct NmsResult {
  xla::XlaOp selected_indices;
  xla::XlaOp num_valid;
};

NmsResult BuildNms(xla::XlaOp boxes, xla::XlaOp scores,
                   xla::XlaOp score_threshold, xla::XlaOp iou_threshold,
                   int64_t output_size);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_NMS_OP_H_