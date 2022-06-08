#include <gtest/gtest.h>

#include "cpp_test_util.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"
#include "torch_xla/csrc/ops/ops.h"
#include "torch_xla/csrc/ops/scalar.h"
#include "torch_xla/csrc/ops/select.h"
#include "torch_xla/csrc/ops/unselect.h"

namespace torch_xla {
namespace cpp_test {

TEST(IrTest, TestScalarCreate) {
  torch::lazy::NodePtr scalar = ScalarOp(1.0, xla::F32);
  ASSERT_TRUE(scalar != nullptr);
}

TEST(IrTest, TestHash) {
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

TEST(IrTest, TestSelectUnselect) {
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

TEST(IrTest, TestScopePusher) {
  torch::lazy::ScopePusher scope("TestScope");
  torch::lazy::NodePtr nodeptr = ScalarOp(1.0, xla::F32);
  auto metaWithScope = nodeptr->metadata();
  EXPECT_EQ(metaWithScope.scope, "TestScope.1");
  EXPECT_EQ(metaWithScope.frame_info.size(), 1);
}

}  // namespace cpp_test
}  // namespace torch_xla
