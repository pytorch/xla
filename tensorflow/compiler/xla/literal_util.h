#pragma once

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/primitive_util.h"

namespace xla {

class LiteralUtil {
 public:
  template <typename NativeT>
  static Literal CreateR0(NativeT value) {
    Literal literal(ShapeUtil::MakeShape(
        primitive_util::NativeToPrimitiveType<NativeT>(), {}));
    literal.Set<NativeT>({}, value);
    return literal;
  }

  template <typename NativeT>
  static Literal CreateR1(absl::Span<const NativeT> values) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  static Literal Zero(PrimitiveType primitive_type) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  static Literal One(PrimitiveType primitive_type) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  static Literal MinValue(PrimitiveType primitive_type) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

}  // namespace xla
