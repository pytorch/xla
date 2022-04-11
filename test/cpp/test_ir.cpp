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
  ir::NodePtr scalar = ir::ops::ScalarOp(1.0, xla::F32);
  ASSERT_TRUE(scalar != nullptr);
}

TEST(IrTest, TestReplace) {
  ir::NodePtr scalar1 = ir::ops::ScalarOp(1.0, xla::F32);
  ir::NodePtr scalar2 = ir::ops::ScalarOp(2.0, xla::F32);
  ir::NodePtr add = ir::Value(scalar1, 0) + ir::Value(scalar2, 0);

  EXPECT_EQ(dynamic_cast<ir::Node*>(scalar1.get())->uses().size(), 1);
  EXPECT_EQ(dynamic_cast<ir::Node*>(scalar2.get())->uses().size(), 1);

  ir::NodePtr scalar3 = ir::ops::ScalarOp(3.0, xla::F32);
  dynamic_cast<ir::Node*>(scalar1.get())->ReplaceAllUsesWith(scalar3);

  EXPECT_EQ(dynamic_cast<ir::Node*>(scalar1.get())->uses().size(), 0);
  EXPECT_EQ(dynamic_cast<ir::Node*>(scalar3.get())->uses().size(), 1);

  dynamic_cast<ir::Node*>(add.get())->ReplaceOperand(0, scalar1);
  EXPECT_EQ(dynamic_cast<ir::Node*>(scalar1.get())->uses().size(), 1);
}

TEST(IrTest, TestHash) {
  ir::NodePtr scalar1 = ir::ops::ScalarOp(1.0, xla::F32);
  ir::NodePtr scalar2 = ir::ops::ScalarOp(2.0, xla::F32);
  ir::Value add1 = ir::Value(scalar1, 0) + ir::Value(scalar2, 0);

  ir::NodePtr scalar3 = ir::ops::ScalarOp(1.0, xla::F32);
  ir::NodePtr scalar4 = ir::ops::ScalarOp(2.0, xla::F32);
  ir::Value add2 = ir::Value(scalar3, 0) + ir::Value(scalar4, 0);

  ir::NodePtr scalar5 = ir::ops::ScalarOp(11.0, xla::F32);
  ir::NodePtr scalar6 = ir::ops::ScalarOp(22.0, xla::F32);
  ir::Value add3 = ir::Value(scalar5, 0) + ir::Value(scalar6, 0);

  EXPECT_EQ(add1->hash(), add2->hash());
  EXPECT_NE(add1->hash(), add3->hash());

  ir::Value sub = ir::Value(scalar1, 0) + ir::Value(scalar2, 0);

  EXPECT_NE(add1->hash(), sub->hash());
}

TEST(IrTest, TestSelectUnselect) {
  ForEachDevice([&](const Device& device) {
    at::Tensor a =
        at::rand({4, 16, 3}, at::TensorOptions(at::kFloat)).abs() + 1.0;

    ir::Value v_a = GetTensorIrValue(a, device);
    ir::Value v_s = ir::MakeNode<ir::ops::Select>(v_a, /*dim=*/1, /*start=*/3,
                                                  /*end=*/14, /*stride=*/3);

    auto results = ExecuteAndFetch({v_s}, device);
    at::Tensor b =
        a.slice(/*dim=*/1, /*start=*/3, /*end=*/14, /*stride=*/3).clone();
    AllClose(results.front(), b);

    // Paste zeros back into the selected view.
    at::Tensor z = at::zeros_like(b);
    ir::Value v_z = GetTensorIrValue(z, device);
    ir::Value v_u =
        ir::MakeNode<ir::ops::Unselect>(v_a, v_z, /*dim=*/1, /*start=*/3,
                                        /*end=*/14, /*stride=*/3);
    results = ExecuteAndFetch({v_u}, device);
    // Fetch back the zeros.
    at::Tensor d = results.front().cpu().slice(/*dim=*/1, /*start=*/3,
                                               /*end=*/14, /*stride=*/3);
    // Sum must be zero as all the other tensor items are >= 1.0.
    EXPECT_EQ(d.sum().item().toDouble(), 0.0);
  });
}

}  // namespace cpp_test
}  // namespace torch_xla
