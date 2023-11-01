#ifndef XLA_TORCH_XLA_CSRC_QUANT_UTIL_H_
#define XLA_TORCH_XLA_CSRC_QUANT_UTIL_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "xla/primitive_util.h"

namespace torch_xla {

struct QuantParams {
  std::vector<float> scale;
  std::vector<float> zero_point;
  int quant_min;
  int quant_max;
  int axis;
  std::string dtype;
  xla::PrimitiveType expressed_type;

  QuantParams(const std::vector<float>& scale,
              const std::vector<float>& zero_point, int quant_min,
              int quant_max, int axis, std::string dtype,
              xla::PrimitiveType expressed_type)
      : scale(scale),
        zero_point(zero_point),
        quant_min(quant_min),
        quant_max(quant_max),
        axis(axis),
        dtype(dtype),
        expressed_type(expressed_type) {}

  std::string SerializeToAttrDictStr() const;
};

std::string SeralizeFloatVector(const std::vector<float>& v,
                                bool append_decimal = false);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_QUANT_UTIL_H_
