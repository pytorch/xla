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

#include "torch/csrc/lazy/core/dynamic_ir.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

/**
 * The goal of "dynamic" Nodes is to patch a hole in our tracing.
 * Previously, if a user called `sizes` on a Tensor, it would leak out
 * of our tracing system, as `sizes` returns a torch.Size or an int. To
 * prevent this from happening, we introduce torch::lazy::DimensionNode, a new
 * type of XlaNode that abstracts the operation of getting the dimensions of a
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

// Represents the result of calling `size` on a Tensor
class SizeNode : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeNode(torch::lazy::Value input, size_t dim);
  int64_t getStaticValue() const override;
  bool isDynamic() const override;
  std::string ToString() const override;
  size_t dim_ = 0;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;
};

class SizeAdd : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeAdd(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getStaticValue() const override;
  bool isDynamic() const override;
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;
};

class SizeMul : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeMul(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getStaticValue() const override;
  bool isDynamic() const override;
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;
};

class SizeDiv : public XlaNode, public torch::lazy::DimensionNode {
 public:
  SizeDiv(torch::lazy::Value a, torch::lazy::Value b);
  int64_t getStaticValue() const override;
  bool isDynamic() const override;
  std::string ToString() const override;
  virtual XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla
