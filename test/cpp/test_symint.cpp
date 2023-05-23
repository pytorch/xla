#include <gtest/gtest.h>

#include <iostream>

#include "test/cpp/cpp_test_util.h"
#include "torch_xla/csrc/LazyIr.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/torch_util.h"
using std::cerr;

namespace torch_xla {
namespace cpp_test {

static c10::SymInt make_symint(const torch::lazy::NodePtr& p) {
  return c10::SymInt(static_cast<c10::SymNode>(
      c10::make_intrusive<XLASymNodeImpl>(p, PyType::INT)));
}

TEST(SymintTest, TestStaticSymint) {
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

TEST(SymintTest, TestStaticSymints) {
  // We have to init a std::vector<int64_t> here. Passing a temp variable to
  // fromIntArrayRef will result in unexpected behavior.
  std::vector<int64_t> sizes = {6, 19, 10};
  c10::SymIntArrayRef static_symints = c10::fromIntArrayRefSlow(sizes);
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
  // This is not a dynamic size from xla perspective but it is a symint that
  // wraps around a SizeNode instead of a scalar.
  c10::SymInt dynamic_symint = make_symint(size_node);
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

TEST(SymintTest, TestSizeConstant) {
  torch::lazy::NodePtr sc10 = torch::lazy::MakeNode<SizeConstant>(10);
  EXPECT_EQ(torch_xla::DimCast(sc10)->getStaticValue(), 10);
  EXPECT_EQ(torch_xla::DimCast(sc10)->getDynamicValue(), 10);
  torch::lazy::NodePtr sc15 = torch::lazy::MakeNode<SizeConstant>(15);
  EXPECT_EQ(torch_xla::DimCast(sc15)->getStaticValue(), 15);
  EXPECT_EQ(torch_xla::DimCast(sc15)->getDynamicValue(), 15);
  torch::lazy::NodePtr add25 = torch::lazy::MakeNode<SizeAdd>(sc10, sc15);
  EXPECT_EQ(torch_xla::DimCast(add25)->getStaticValue(), 25);
  EXPECT_EQ(torch_xla::DimCast(add25)->getDynamicValue(),
            torch_xla::DimCast(add25)->getStaticValue());

  torch::lazy::Value scalar_value =
      torch::lazy::Value(ScalarOp(1.0, xla::F32), 0);

  std::vector<int64_t> target_size = {9};
  torch::lazy::NodePtr expand_node =
      torch::lazy::MakeNode<Expand>(scalar_value, target_size);
  torch::lazy::Value expand_value = torch::lazy::Value(expand_node, 0);

  torch::lazy::NodePtr size_node =
      torch::lazy::MakeNode<SizeNode>(expand_value, /*dim=*/0);

  torch::lazy::NodePtr add19 = torch::lazy::MakeNode<SizeAdd>(sc10, size_node);
  EXPECT_EQ(torch_xla::DimCast(add19)->getStaticValue(), 19);
  EXPECT_EQ(torch_xla::DimCast(add19)->getDynamicValue(),
            torch_xla::DimCast(add19)->getStaticValue());
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
    // This is not a dynamic size from xla perspective but it is a symint that
    // wraps around a SizeNode instead of a scalar.
    dynamic_symints.push_back(make_symint(size_node));
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

TEST(SymintTest, TestDynamicSymintArithmetic) {
  torch::lazy::Value scalar_value =
      torch::lazy::Value(ScalarOp(1.0, xla::F32), 0);

  std::vector<int64_t> target_size = {10, 20, 30};
  torch::lazy::NodePtr expand_node =
      torch::lazy::MakeNode<Expand>(scalar_value, target_size);
  torch::lazy::Value expand_value = torch::lazy::Value(expand_node, 0);

  torch::lazy::NodePtr abs_node = torch::lazy::MakeNode<Abs>(expand_value);
  torch::lazy::NodePtr relu_node = torch::lazy::MakeNode<Relu>(expand_value);

  torch::lazy::NodePtr size_abs_node = torch::lazy::MakeNode<SizeNode>(
      torch::lazy::Value{abs_node, 0}, /*dim=*/0);
  torch::lazy::NodePtr size_relu_node = torch::lazy::MakeNode<SizeNode>(
      torch::lazy::Value{relu_node, 0}, /*dim=*/0);

  c10::SymInt a = make_symint(size_abs_node);
  c10::SymInt b = make_symint(size_relu_node);

  // Testing XLASymNodeImpl::add
  c10::SymInt c = a + b;
  auto size_add_symnode =
      dynamic_cast<XLASymNodeImpl*>(c.toSymNodeImplUnowned());
  ASSERT_TRUE(size_add_symnode);
  auto size_add =
      std::dynamic_pointer_cast<torch_xla::SizeAdd>(size_add_symnode->node());
  ASSERT_EQ(size_add->operands().at(0).node, size_abs_node.get());
  ASSERT_EQ(size_add->operands().at(1).node, size_relu_node.get());

  // Testing XLASymNodeImpl::mul
  c = a * b;
  auto size_mul_symnode =
      dynamic_cast<XLASymNodeImpl*>(c.toSymNodeImplUnowned());
  ASSERT_TRUE(size_mul_symnode);
  auto size_mul =
      std::dynamic_pointer_cast<torch_xla::SizeMul>(size_mul_symnode->node());
  ASSERT_TRUE(size_mul);
  ASSERT_EQ(size_mul->operands().at(0).node, size_abs_node.get());
  ASSERT_EQ(size_mul->operands().at(1).node, size_relu_node.get());

  // Testing XLASymNodeImpl::floordiv
  c = a / b;
  auto size_floordiv_symnode =
      dynamic_cast<XLASymNodeImpl*>(c.toSymNodeImplUnowned());
  ASSERT_TRUE(size_floordiv_symnode);
  auto size_floordiv =
      std::dynamic_pointer_cast<SizeDiv>(size_floordiv_symnode->node());
  ASSERT_TRUE(size_floordiv);
  ASSERT_EQ(size_floordiv->operands().at(0).node, size_abs_node.get());
  ASSERT_EQ(size_floordiv->operands().at(1).node, size_relu_node.get());
}

TEST(SymintTest, TestXLASymNodeImplStr) {
  torch::lazy::Value scalar = torch::lazy::Value(ScalarOp(1.0, xla::F32), 0);
  std::vector<int64_t> shape = {2, 3, 4};
  torch::lazy::NodePtr expand_node =
      torch::lazy::MakeNode<Expand>(scalar, shape);
  torch::lazy::Value expand_value = torch::lazy::Value(expand_node, 0);
  torch::lazy::NodePtr size_node =
      torch::lazy::MakeNode<SizeNode>(expand_value, 0);
  c10::SymNode symint_node =
      c10::make_intrusive<XLASymNodeImpl>(size_node, PyType::INT);
  ASSERT_EQ(symint_node.get()->str(), "<=2");
}

}  // namespace cpp_test
}  // namespace torch_xla
