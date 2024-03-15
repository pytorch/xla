#include <gtest/gtest.h>

#include <stdexcept>

#include "test/cpp/cpp_test_util.h"
#include "test/cpp/torch_xla_test.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/dynamic_ir.h"
#include "torch_xla/csrc/ops/expand.h"
#include "torch_xla/csrc/ops/nonzero.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/unselect.h"
#include "torch_xla/csrc/ops/update_slice.h"

namespace torch_xla {
namespace cpp_test {

class IrTest : public TorchXlaTest {};

TEST_F(IrTest, TestScalarCreate) {
  torch::lazy::NodePtr scalar = ScalarOp(1.0, xla::F32);
  ASSERT_TRUE(scalar != nullptr);
}

TEST_F(IrTest, TestHash) {
  torch::lazy::NodePtr scalar1 = ScalarOp(1.0, xla::F32);
  torch::lazy::NodePtr scalar2 = ScalarOp(2.0, xla::F32);
  torch::lazy::Value add1 =
      torch::lazy::Value(scalar1, 0) + torch::lazy::Value(scalar2, 0);

  torch::lazy::NodePtr scalar3 = ScalarOp(1.0, xla::F32);
  torch::lazy::NodePtr scalar4 = ScalarOp(2.0, xla::F32);
  torch::lazy::Value add2 =
      torch::lazy::Value(scalar3, 0) + torch::lazy::Value(scalar4, 0);

  torch::lazy::NodePtr scalar5 = ScalarOp(11.0, xla::F32);
  torch::lazy::NodePtr scalar6 = ScalarOp(22.0, xla::F32);
  torch::lazy::Value add3 =
      torch::lazy::Value(scalar5, 0) + torch::lazy::Value(scalar6, 0);

  EXPECT_EQ(add1->hash(), add2->hash());
  EXPECT_NE(add1->hash(), add3->hash());

  torch::lazy::Value sub =
      torch::lazy::Value(scalar1, 0) - torch::lazy::Value(scalar2, 0);

  EXPECT_NE(add1->hash(), sub->hash());
}

TEST_F(IrTest, TestSelectUnselect) {
  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    at::Tensor a =
        at::rand({4, 16, 3}, at::TensorOptions(at::kFloat)).abs() + 1.0;

    torch::lazy::Value v_a = GetTensorIrValue(a, device);
    torch::lazy::Value v_s =
        torch::lazy::MakeNode<Select>(v_a, /*dim=*/1, /*start=*/3,
                                      /*end=*/14, /*stride=*/3);

    auto results = ExecuteAndFetch({v_s}, device);
    at::Tensor b =
        a.slice(/*dim=*/1, /*start=*/3, /*end=*/14, /*stride=*/3).clone();
    AllClose(results.front(), b);

    // Paste zeros back into the selected view.
    at::Tensor z = at::zeros_like(b);
    torch::lazy::Value v_z = GetTensorIrValue(z, device);
    torch::lazy::Value v_u =
        torch::lazy::MakeNode<Unselect>(v_a, v_z, /*dim=*/1, /*start=*/3,
                                        /*end=*/14, /*stride=*/3);
    results = ExecuteAndFetch({v_u}, device);
    // Fetch back the zeros.
    at::Tensor d = results.front().cpu().slice(/*dim=*/1, /*start=*/3,
                                               /*end=*/14, /*stride=*/3);
    // Sum must be zero as all the other tensor items are >= 1.0.
    EXPECT_EQ(d.sum().item().toDouble(), 0.0);
  });
}

TEST_F(IrTest, TestScopePusherWithoutDebugging) {
  bool restore_FLAGS_torch_lazy_ir_debug = FLAGS_torch_lazy_ir_debug;
  FLAGS_torch_lazy_ir_debug = false;
  torch::lazy::ScopePusher scope("TestScope");
  torch::lazy::NodePtr nodeptr = ScalarOp(1.0, xla::F32);
  auto metaWithScope = nodeptr->metadata();
  EXPECT_EQ(metaWithScope.scope, "");
  EXPECT_EQ(metaWithScope.frame_info.size(), 0);
  FLAGS_torch_lazy_ir_debug = restore_FLAGS_torch_lazy_ir_debug;
}

TEST_F(IrTest, TestScopePusherWithDebugging) {
  bool restore_FLAGS_torch_lazy_ir_debug = FLAGS_torch_lazy_ir_debug;
  FLAGS_torch_lazy_ir_debug = true;
  torch::lazy::ScopePusher scope("TestScope");
  torch::lazy::NodePtr nodeptr = ScalarOp(1.0, xla::F32);
  auto metaWithScope = nodeptr->metadata();
  ASSERT_TRUE(metaWithScope.scope.find("TestScope") != std::string::npos);
  EXPECT_EQ(metaWithScope.frame_info.size(), 1);
  FLAGS_torch_lazy_ir_debug = restore_FLAGS_torch_lazy_ir_debug;
}

TEST_F(IrTest, TestSizeNode) {
  torch::lazy::NodePtr scalar_node =
      ScalarOp(1.0, xla::ShapeUtil::MakeShape(xla::F32, {3, 4}));
  torch::lazy::NodePtr size_node_0 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 0);
  torch::lazy::NodePtr size_node_1 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 1);
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_0 =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node_0);
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_1 =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node_1);

  EXPECT_EQ(dim_node_0->getStaticValue(), 3);
  EXPECT_EQ(dim_node_0->getDynamicValue(), 3);
  EXPECT_EQ(dim_node_1->getStaticValue(), 4);
  EXPECT_EQ(dim_node_1->getDynamicValue(), 4);

  // calling getDynamicValue the second time should use the cached value.
  EXPECT_EQ(dim_node_0->getDynamicValue(), 3);
  EXPECT_EQ(dim_node_1->getDynamicValue(), 4);
  ExpectCounterChanged("CachedSizeNodeValue", cpp_test::GetIgnoredCounters());

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    // Lower the SizeNode and execute the GetDimensionSize.
    auto results = ExecuteAndFetch({size_node_0, size_node_1}, device);
    EXPECT_EQ(results[0].sum().item().toInt(), 3);
    EXPECT_EQ(results[1].sum().item().toInt(), 4);
  });
}

TEST_F(IrTest, TestSizeNodeDynamic) {
  int64_t num_non_zero_element = 1;
  int64_t num_row = 10;
  int64_t num_col = 10;
  torch::lazy::NodePtr nonzero_node =
      CreateNonZeroNode2d(num_non_zero_element, num_row, num_col);

  torch::lazy::NodePtr size_node_nonzero_0 =
      torch::lazy::MakeNode<SizeNode>(nonzero_node, 0);
  EXPECT_EQ(size_node_nonzero_0->ToString(), "aten::size");
  torch::lazy::NodePtr size_node_nonzero_1 =
      torch::lazy::MakeNode<SizeNode>(nonzero_node, 1);
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_0 =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(
          size_node_nonzero_0);
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_1 =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(
          size_node_nonzero_1);

  EXPECT_EQ(dim_node_0->getStaticValue(), 100);
  EXPECT_EQ(dim_node_0->getDynamicValue(), num_non_zero_element);
  EXPECT_EQ(dim_node_1->getStaticValue(), 2);
  EXPECT_EQ(dim_node_1->getDynamicValue(), 2);

  // calling getDynamicValue the second time should use the cached value.
  EXPECT_EQ(dim_node_0->getDynamicValue(), 1);
  EXPECT_EQ(dim_node_1->getDynamicValue(), 2);
  ExpectCounterChanged("CachedSizeNodeValue", cpp_test::GetIgnoredCounters());
}

TEST_F(IrTest, TestSizeAddNode) {
  torch::lazy::NodePtr scalar_node =
      ScalarOp(1.0, xla::ShapeUtil::MakeShape(xla::F32, {3, 4}));
  torch::lazy::NodePtr size_node_0 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 0);
  torch::lazy::NodePtr size_node_1 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 1);
  torch::lazy::NodePtr size_node_add =
      torch::lazy::MakeNode<SizeAdd>(size_node_0, size_node_1);
  EXPECT_EQ(size_node_add->ToString(), "aten::size_add");
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_add =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node_add);

  EXPECT_EQ(dim_node_add->getStaticValue(), 7);
  EXPECT_EQ(dim_node_add->getDynamicValue(), 7);

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    // Lower the SizeAddNode and execute the GetDimensionSize.
    auto results = ExecuteAndFetch({size_node_add}, device);
    EXPECT_EQ(results[0].sum().item().toInt(), 7);
  });
}

TEST_F(IrTest, TestSizeAddNodeDynamicOnSameTensor) {
  int64_t num_non_zero_element = 1;
  int64_t num_row = 10;
  int64_t num_col = 10;
  torch::lazy::NodePtr nonzero_node =
      CreateNonZeroNode2d(num_non_zero_element, num_row, num_col);
  torch::lazy::Value node_with_dynamism = torch::lazy::Value(nonzero_node, 0);

  // static value = 100, dynamic value = 1
  torch::lazy::NodePtr size_node_nonzero_0 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism, 0);
  // static value = 2, dynamic value = 2
  torch::lazy::NodePtr size_node_nonzero_1 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism, 1);

  torch::lazy::NodePtr node_add = torch::lazy::MakeNode<SizeAdd>(
      torch::lazy::Value(size_node_nonzero_0, 0),
      torch::lazy::Value(size_node_nonzero_1, 0));

  std::shared_ptr<torch::lazy::DimensionNode> dim_node_add =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(node_add);
  EXPECT_EQ(dim_node_add->getStaticValue(), 102);
  EXPECT_EQ(dim_node_add->getDynamicValue(), 3);
}

TEST_F(IrTest, TestSizeAddNodeDynamicOnDifferentTensor) {
  int64_t num_non_zero_element = 1;
  int64_t num_row = 10;
  int64_t num_col = 10;
  torch::lazy::NodePtr nonzero_node_0 =
      CreateNonZeroNode2d(num_non_zero_element, num_row, num_col);
  torch::lazy::Value node_with_dynamism_0 =
      torch::lazy::Value(nonzero_node_0, 0);
  torch::lazy::NodePtr nonzero_node_1 =
      CreateNonZeroNode2d(num_non_zero_element, num_row, num_col);
  torch::lazy::Value node_with_dynamism_1 =
      torch::lazy::Value(nonzero_node_1, 0);

  // static value = 100, dynamic value = 1
  torch::lazy::NodePtr size_node_nonzero_0 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism_0, 0);
  // static value = 100, dynamic value = 1
  torch::lazy::NodePtr size_node_nonzero_1 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism_1, 0);

  torch::lazy::NodePtr node_add = torch::lazy::MakeNode<SizeAdd>(
      torch::lazy::Value(size_node_nonzero_0, 0),
      torch::lazy::Value(size_node_nonzero_1, 0));

  std::shared_ptr<torch::lazy::DimensionNode> dim_node_add =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(node_add);
  EXPECT_EQ(dim_node_add->getStaticValue(), 200);
  EXPECT_EQ(dim_node_add->getDynamicValue(), 2);
}

TEST_F(IrTest, TestSizeMulNode) {
  torch::lazy::NodePtr scalar_node =
      ScalarOp(1.0, xla::ShapeUtil::MakeShape(xla::F32, {3, 4}));
  torch::lazy::NodePtr size_node_0 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 0);
  torch::lazy::NodePtr size_node_1 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 1);
  torch::lazy::NodePtr size_node_mul =
      torch::lazy::MakeNode<SizeMul>(size_node_0, size_node_1);
  EXPECT_EQ(size_node_mul->ToString(), "aten::size_mul");
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_mul =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node_mul);

  EXPECT_EQ(dim_node_mul->getStaticValue(), 12);
  EXPECT_EQ(dim_node_mul->getDynamicValue(), 12);

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    // Lower the SizeAddNode and execute the GetDimensionSize.
    auto results = ExecuteAndFetch({size_node_mul}, device);
    EXPECT_EQ(results[0].sum().item().toInt(), 12);
  });
}

TEST_F(IrTest, TestSizeMulNodeDynamic) {
  int64_t num_non_zero_element = 1;
  int64_t num_row = 10;
  int64_t num_col = 10;
  torch::lazy::NodePtr nonzero_node =
      CreateNonZeroNode2d(num_non_zero_element, num_row, num_col);
  torch::lazy::Value node_with_dynamism = torch::lazy::Value(nonzero_node, 0);

  // static value = 100, dynamic value = 1
  torch::lazy::NodePtr size_node_nonzero_0 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism, 0);
  // static value = 2, dynamic value = 2
  torch::lazy::NodePtr size_node_nonzero_1 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism, 1);

  torch::lazy::NodePtr node_mul = torch::lazy::MakeNode<SizeMul>(
      torch::lazy::Value(size_node_nonzero_0, 0),
      torch::lazy::Value(size_node_nonzero_1, 0));

  std::shared_ptr<torch::lazy::DimensionNode> dim_node_mul =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(node_mul);
  EXPECT_EQ(dim_node_mul->getStaticValue(), 200);
  EXPECT_EQ(dim_node_mul->getDynamicValue(), 2);
}

TEST_F(IrTest, TestSizeDivNode) {
  torch::lazy::NodePtr scalar_node =
      ScalarOp(1.0, xla::ShapeUtil::MakeShape(xla::F32, {12, 5}));
  torch::lazy::NodePtr size_node_0 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 0);
  torch::lazy::NodePtr size_node_1 =
      torch::lazy::MakeNode<SizeNode>(scalar_node, 1);
  torch::lazy::NodePtr size_node_div =
      torch::lazy::MakeNode<SizeDiv>(size_node_0, size_node_1);
  EXPECT_EQ(size_node_div->ToString(), "aten::size_div");
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_div =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(size_node_div);

  EXPECT_EQ(dim_node_div->getStaticValue(), 2);
  EXPECT_EQ(dim_node_div->getDynamicValue(), 2);

  ForEachDevice([&](const torch::lazy::BackendDevice& device) {
    // Lower the SizeAddNode and execute the GetDimensionSize.
    auto results = ExecuteAndFetch({size_node_div}, device);
    EXPECT_EQ(results[0].sum().item().toInt(), 2);
  });
}

TEST_F(IrTest, TestSizeDivNodeDynamic) {
  int64_t num_non_zero_element = 1;
  int64_t num_row = 10;
  int64_t num_col = 10;
  torch::lazy::NodePtr nonzero_node =
      CreateNonZeroNode2d(num_non_zero_element, num_row, num_col);
  torch::lazy::Value node_with_dynamism = torch::lazy::Value(nonzero_node, 0);

  // static value = 100, dynamic value = 1
  torch::lazy::NodePtr size_node_nonzero_0 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism, 0);
  // static value = 2, dynamic value = 2
  torch::lazy::NodePtr size_node_nonzero_1 =
      torch::lazy::MakeNode<SizeNode>(node_with_dynamism, 1);

  torch::lazy::NodePtr node_div = torch::lazy::MakeNode<SizeDiv>(
      torch::lazy::Value(size_node_nonzero_0, 0),
      torch::lazy::Value(size_node_nonzero_1, 0));

  std::shared_ptr<torch::lazy::DimensionNode> dim_node_div =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(node_div);
  EXPECT_EQ(dim_node_div->getStaticValue(), 50);
  EXPECT_EQ(dim_node_div->getDynamicValue(), 0);
}

TEST_F(IrTest, TestSizeDivNodeDynamicByZero) {
  torch::lazy::Value scalar_value =
      torch::lazy::Value(ScalarOp(0.0, xla::F32), 0);
  std::vector<int64_t> target_size = {2, 2};
  torch::lazy::Value node = torch::lazy::Value(
      torch::lazy::MakeNode<Expand>(scalar_value, target_size));
  torch::lazy::NodePtr nonzero_node = torch::lazy::MakeNode<NonZero>(node);

  // static value = 4, dynamic value = 0
  torch::lazy::NodePtr size_node_nonzero_0 =
      torch::lazy::MakeNode<SizeNode>(nonzero_node, 0);
  // static value = 2, dynamic value = 2
  torch::lazy::NodePtr size_node_nonzero_1 =
      torch::lazy::MakeNode<SizeNode>(nonzero_node, 1);

  torch::lazy::NodePtr node_div = torch::lazy::MakeNode<SizeDiv>(
      torch::lazy::Value(size_node_nonzero_1, 0),
      torch::lazy::Value(size_node_nonzero_0, 0));
  std::shared_ptr<torch::lazy::DimensionNode> dim_node_div =
      std::dynamic_pointer_cast<torch::lazy::DimensionNode>(node_div);

  EXPECT_THROW(dim_node_div->getDynamicValue(), std::runtime_error);
}

}  // namespace cpp_test
}  // namespace torch_xla
