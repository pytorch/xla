#pragma once

// #include <torch/ATen/core/symbol.h>

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "torch_xla/csrc/ir.h"
// #include <torch/c10/core/ScalarType.h>
// #include <torch/csrc/lazy/core/hash.h>
// #include <torch/csrc/lazy/core/ir.h>
// #include <torch/csrc/lazy/core/ir_metadata.h>
// #include <torch/csrc/lazy/ts_backend/ts_node.h>
// #include <torch/c10/util/Flags.h>

namespace torch_xla {
namespace ir {
namespace ops {

/**
 * The goal of "dynamic" Nodes is to patch a hole in our tracing.
 * Previously, if a user called `sizes` on a Tensor, it would leak out
 * of our tracing system, as `sizes` returns a torch.Size or an int. To
 * prevent this from happening, we introduce DimensionNode, a new type
 * of Node that abstracts the operation of getting the dimensions of a
 * Tensor.
 *
 * Consider the following example:
 * ```
 * numel = x.shape()[0] * x.shape()[1]
 * ```
 *
 * Here, `x.shape()[i]` will be a SizeNode (subclass of DimensionNode),
 * and the multiplication of the two SizeNodes will be represented by
 * a SizeMul (also a subclass of DimensionNode). Through this, we can
 * prevent `numel` from being represented as a Python int and thus
 * burned into the Graph.
 */

class DimensionNode : public Node {
 public:
  // DimensionNode(torch::lazy::OpKind op, OpList operands, torch::lazy:hash_t hash_seed = kHashSeed);
  DimensionNode(torch::lazy::OpKind op, OpList operands, torch::lazy::hash_t hash_seed);
  bool isDynamic() {
      return false;
  }

  std::string ToString() const override;

  virtual int64_t getStaticValue() const = 0;
};

// Represents the result of calling `size` on a Tensor
class SizeNode : public DimensionNode {
 public:
  SizeNode(Value input, size_t dim);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
  size_t dim_ = 0;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;
};

class SizeAdd: public DimensionNode {
 public:
  SizeAdd(Value a, Value b);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
};

class SizeMul: public DimensionNode {
 public:
  SizeMul(Value a, Value b);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
};

class SizeDiv: public DimensionNode {
 public:
  SizeDiv(Value a, Value b);
  int64_t getStaticValue() const override;
  std::string ToString() const override;
};


}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
