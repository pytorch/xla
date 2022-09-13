#include <gtest/gtest.h>

#include <iostream>

#include "cpp_test_util.h"
#include "torch_xla/csrc/generated/LazyIr.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
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

  std::vector<torch::lazy::NodePtr> size_nodes = si_element.GetSizeNodes();
  // Static SymIntElements should not have size_node
  EXPECT_EQ(size_nodes.size(), 1);
  EXPECT_EQ(size_nodes[0], nullptr);
  EXPECT_EQ(si_element.GetSizeNode(0), nullptr);
}

TEST(SymintTest, TestSaticSymints) {
  // We have to init a std::vector<int64_t> here. Passing a temp variable to
  // fromIntArrayRef will result in unexpected behavior.
  std::vector<int64_t> sizes = {6, 19, 10};
  c10::SymIntArrayRef static_symints = c10::fromIntArrayRef(sizes);
  SymIntElements si_element(static_symints);

  std::vector<int64_t> upper_bound = si_element.GetUpperBounds();
  EXPECT_EQ(upper_bound.size(), 3);
  EXPECT_EQ(upper_bound, std::vector<int64_t>({6, 19, 10}));

  std::vector<bool> dynamic_dims = si_element.GetDynamicDims();
  EXPECT_EQ(dynamic_dims.size(), 3);
  EXPECT_EQ(dynamic_dims, std::vector<bool>({false, false, false}));

  std::vector<torch::lazy::NodePtr> size_nodes = si_element.GetSizeNodes();
  // Static SymIntElements should not have size_node
  EXPECT_EQ(size_nodes.size(), 3);
  EXPECT_EQ(size_nodes,
            std::vector<torch::lazy::NodePtr>({nullptr, nullptr, nullptr}));
  EXPECT_EQ(si_element.GetSizeNode(0), nullptr);
}

TEST(SymintTest, TestDynamicSymint) {
  torch::lazy::Value scalar_value =
      torch::lazy::Value(ScalarOp(1.0, xla::F32), 0);
  std::vector<int64_t> target_size = {2, 3, 5};
  torch::lazy::NodePtr expand_node =
      torch::lazy::MakeNode<Expand>(scalar_value, target_size);
  torch::lazy::Value expand_value = torch::lazy::Value(expand_node, 0);
  torch::lazy::NodePtr size_node =
      torch::lazy::MakeNode<SizeNode>(expand_value, /*dim=*/0);
  auto symint_node =
      c10::make_intrusive<torch::lazy::SymIntNodeImpl>(size_node);
  // This is not a dynamic size from xla perspective but it is a symint that
  // wraps around a SizeNode instead of a scalar.
  c10::SymInt dynamic_symint = symint_node->toSymInt();
  SymIntElements si_element(dynamic_symint);

  std::vector<int64_t> upper_bound = si_element.GetUpperBounds();
  EXPECT_EQ(upper_bound.size(), 1);
  EXPECT_EQ(upper_bound[0], 2);

  std::vector<bool> dynamic_dims = si_element.GetDynamicDims();
  EXPECT_EQ(dynamic_dims.size(), 1);
  EXPECT_EQ(dynamic_dims[0], true);

  std::vector<torch::lazy::NodePtr> size_nodes = si_element.GetSizeNodes();
  EXPECT_EQ(size_nodes.size(), 1);
  EXPECT_EQ(si_element.GetSizeNode(0), size_node);
}

TEST(SymintTest, TestDynamicSymints) {
  torch::lazy::Value scalar_value =
      torch::lazy::Value(ScalarOp(1.0, xla::F32), 0);
  std::vector<int64_t> target_size = {2, 3, 5};
  torch::lazy::NodePtr expand_node =
      torch::lazy::MakeNode<Expand>(scalar_value, target_size);
  torch::lazy::Value expand_value = torch::lazy::Value(expand_node, 0);
  std::vector<c10::SymInt> dynamic_symints;
  std::vector<torch::lazy::NodePtr> size_nodes;
  for (int i = 0; i < 3; i++) {
    torch::lazy::NodePtr size_node =
        torch::lazy::MakeNode<SizeNode>(expand_value, /*dim=*/i);
    size_nodes.push_back(size_node);
    auto symint_node =
        c10::make_intrusive<torch::lazy::SymIntNodeImpl>(size_node);
    // This is not a dynamic size from xla perspective but it is a symint that
    // wraps around a SizeNode instead of a scalar.
    dynamic_symints.push_back(symint_node->toSymInt());
  }

  c10::SymIntArrayRef ref(dynamic_symints);
  SymIntElements si_element(ref);

  std::vector<int64_t> upper_bound = si_element.GetUpperBounds();
  EXPECT_EQ(upper_bound.size(), 3);
  EXPECT_EQ(upper_bound, std::vector<int64_t>({2, 3, 5}));

  std::vector<bool> dynamic_dims = si_element.GetDynamicDims();
  EXPECT_EQ(dynamic_dims.size(), 3);
  EXPECT_EQ(dynamic_dims, std::vector<bool>({true, true, true}));

  std::vector<torch::lazy::NodePtr> si_element_size_nodes =
      si_element.GetSizeNodes();
  EXPECT_EQ(si_element_size_nodes.size(), 3);
  EXPECT_EQ(si_element_size_nodes, size_nodes);
}

}  // namespace cpp_test
}  // namespace torch_xla
