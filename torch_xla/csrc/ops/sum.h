#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "tensorflow/core/lib/gtl/array_slice.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Sum : public Node {
 public:
  Sum(const Value& input, std::vector<xla::int64> dimensions,
      bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<xla::int64>& dimensions() const { return dimensions_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  std::vector<xla::int64> dimensions_;
  bool keep_reduced_dimensions_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
