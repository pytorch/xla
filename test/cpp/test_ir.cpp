#include <gtest/gtest.h>

#include "cpp_test_util.h"
#include "ir.h"
#include "lowering_context.h"
#include "ops/arithmetic_ir_ops.h"
#include "ops/ops.h"
#include "ops/scalar.h"

namespace torch_xla {
namespace cpp_test {

TEST(IrTest, TestScalarCreate) {
  ir::NodePtr scalar = ir::ops::ScalarOp(1.0, xla::F32);
  ASSERT_TRUE(scalar != nullptr);
}

TEST(IrTest, TestReplace) {
  ir::NodePtr scalar1 = ir::ops::ScalarOp(1.0, xla::F32);
  ir::NodePtr scalar2 = ir::ops::ScalarOp(2.0, xla::F32);
  ir::NodeOperand add = scalar1 + scalar2;
  EXPECT_EQ(scalar1->uses().size(), 1);
  EXPECT_EQ(scalar2->uses().size(), 1);

  ir::NodePtr scalar3 = ir::ops::ScalarOp(3.0, xla::F32);
  scalar1->ReplaceAllUsesWith(scalar3);
  EXPECT_EQ(scalar1->uses().size(), 0);
  EXPECT_EQ(scalar3->uses().size(), 1);
}

}  // namespace cpp_test
}  // namespace torch_xla
