#pragma once

#include <string>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {

class DumpUtil {
 public:
  static std::string ToDot(
      tensorflow::gtl::ArraySlice<const Node* const> nodes);

  static std::string PostOrderToDot(
      tensorflow::gtl::ArraySlice<const Node* const> post_order,
      tensorflow::gtl::ArraySlice<const Node* const> roots);

  static std::string ToText(
      tensorflow::gtl::ArraySlice<const Node* const> nodes);

  static std::string PostOrderToText(
      tensorflow::gtl::ArraySlice<const Node* const> post_order,
      tensorflow::gtl::ArraySlice<const Node* const> roots);

  static std::string ToHlo(tensorflow::gtl::ArraySlice<const Value> values);
};

}  // namespace ir
}  // namespace torch_xla
