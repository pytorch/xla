#include <gtest/gtest.h>

#include <iostream>

#include "cpp_test_util.h"
#include "torch_xla/csrc/generated/LazyIr.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/torch_util.h"
using std::cerr;

namespace torch_xla {
namespace cpp_test {

TEST(SymintTest, TestSaticSymint) {
  c10::SymInt static_symint(5);
  SymIntElements si_element(static_symint);

  std::vector<int64_t> upper_bound = si_element.GetUpperBounds();
  EXPECT_EQ(upper_bound.size(), 1);
  EXPECT_EQ(upper_bound[0], 5);

  std::vector<bool> dynamic_dims = si_element.GetDynamicDims();
  EXPECT_EQ(dynamic_dims.size(), 1);
  EXPECT_EQ(dynamic_dims[0], false);

  std::unordered_map<size_t, torch::lazy::NodePtr> size_node_map =
      si_element.GetNodeMap();
  // Static SymIntElements should not have size_node
  EXPECT_EQ(size_node_map.size(), 0);
}

TEST(SymintTest, TestSaticSymints) {
  // We have to init a std::vector<int64_t> here. Passing a temp variable to
  // fromIntArrayRef will result in unexpected behavior.
  std::vector<int64_t> sizes = {6, 19, 10};
  c10::SymIntArrayRef static_symints =
      c10::SymIntArrayRef::fromIntArrayRef(sizes);
  SymIntElements si_element(static_symints);

  std::vector<int64_t> upper_bound = si_element.GetUpperBounds();
  EXPECT_EQ(upper_bound.size(), 3);
  EXPECT_EQ(upper_bound, std::vector<int64_t>({6, 19, 10}));

  std::vector<bool> dynamic_dims = si_element.GetDynamicDims();
  EXPECT_EQ(dynamic_dims.size(), 3);
  EXPECT_EQ(dynamic_dims, std::vector<bool>({false, false, false}));

  std::unordered_map<size_t, torch::lazy::NodePtr> size_node_map =
      si_element.GetNodeMap();
  // Static SymIntElements should not have size_node
  EXPECT_EQ(size_node_map.size(), 0);
}

TEST(SymintTest, TestDynamicSymint) {
  torch::lazy::Value scalar_value =
      torch::lazy::Value(ScalarOp(1.0, xla::F32), 0);
  // Manully assign the torch::lazy::shape to avoid calling shape fn in this
  // test. Note that we have to use one of those codegen ops so they take
  // lazy::shape in constructor.
  std::vector<torch::lazy::Shape> abs_lazy_shapes = {
      torch::lazy::Shape(torch::kFloat, {1})};
  torch::lazy::NodePtr abs_node =
      torch::lazy::MakeNode<Abs>(scalar_value, std::move(abs_lazy_shapes));
  torch::lazy::Value abs_value = torch::lazy::Value(abs_node, 0);
  torch::lazy::NodePtr size_node =
      torch::lazy::MakeNode<SizeNode>(abs_value, /*dim=*/0);
  auto symint_node =
      c10::make_intrusive<torch::lazy::SymIntNodeImpl>(size_node);
  // This is not really a dynamic size per say but it is a symint that wraps
  // around a SizeNode instead of a scalar.
  c10::SymInt dynamic_symint = symint_node->toSymInt();

  SymIntElements si_element(dynamic_symint);

  std::vector<int64_t> upper_bound = si_element.GetUpperBounds();
  EXPECT_EQ(upper_bound.size(), 1);
  EXPECT_EQ(upper_bound[0], 1);

  std::vector<bool> dynamic_dims = si_element.GetDynamicDims();
  EXPECT_EQ(dynamic_dims.size(), 1);
  EXPECT_EQ(dynamic_dims[0], true);

  std::unordered_map<size_t, torch::lazy::NodePtr> size_node_map =
      si_element.GetNodeMap();
  EXPECT_EQ(size_node_map.size(), 1);
  // look up the SizeNode for dimension 0
  EXPECT_TRUE(si_element.GetNode(0) != nullptr);
}

}  // namespace cpp_test
}  // namespace torch_xla