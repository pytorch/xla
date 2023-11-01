#include "torch_xla/csrc/quant_util.h"

#include <iomanip>
#include <iostream>
#include <unordered_map>

#include "torch_xla/csrc/runtime/stablehlo_helper.h"

namespace torch_xla {

std::string QuantParams::SerializeToAttrDictStr() const {
  std::stringstream ss;
  ss << "{";
  if (axis != -1) {
    ss << "quantization_dimension=" << axis << ',';
  }
  ss << "scale=" << SeralizeFloatVector(scale, true) << ',';
  ss << "zero_point=" << SeralizeFloatVector(zero_point) << ',';
  ss << "storage_type=" << GetTorchDtypeToStablehloDtypeMap().at(dtype) << ',';
  ss << "expressed_type=" << GetHloDtypeToStablehloDtypeMap().at(expressed_type)
     << ',';
  ss << "storage_min=" << quant_min << ',';
  ss << "storage_max=" << quant_max;
  ss << '}';
  return ss.str();
}

static inline std::string MaybeAppendDecimalForInteger(float v) {
  std::stringstream ss;
  if (static_cast<int>(v) == v) {
    ss << std::fixed << std::setprecision(2);
  }
  ss << v;
  return ss.str();
}

std::string SeralizeFloatVector(const std::vector<float>& v,
                                bool append_decimal) {
  std::stringstream ss;
  ss << '[';
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) {
      ss << ',';
    }
    if (append_decimal) {
      ss << MaybeAppendDecimalForInteger(v.at(i));
    } else {
      ss << v.at(i);
    }
  }
  ss << ']';
  return ss.str();
}

}  // namespace torch_xla
