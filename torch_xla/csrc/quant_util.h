#ifndef XLA_TORCH_XLA_CSRC_QUANT_UTIL_H_
#define XLA_TORCH_XLA_CSRC_QUANT_UTIL_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "xla/primitive_util.h"

namespace torch_xla {

// Struct for quantization parameters, for per-tensor/channel quant/dequant ops.
struct QuantParams {
  std::vector<float> scale;
  std::vector<int> zero_point;
  int quant_min;
  int quant_max;
  int axis;
  std::string dtype;
  xla::PrimitiveType expressed_type;

  QuantParams(const std::vector<float>& scale,
              const std::vector<int>& zero_point, int quant_min, int quant_max,
              int axis, std::string dtype, xla::PrimitiveType expressed_type)
      : scale(scale),
        zero_point(zero_point),
        quant_min(quant_min),
        quant_max(quant_max),
        axis(axis),
        dtype(dtype),
        expressed_type(expressed_type) {}

  // TODO(lsy323): Remove when qdtype is added in XLA.
  std::string SerializeToAttrDictStr() const;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_QUANT_UTIL_H_
