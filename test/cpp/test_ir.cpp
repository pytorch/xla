#include <gtest/gtest.h>

#include "ir.h"
#include "ops/ops.h"
#include "ops/scalar.h"

TEST(IrTest, TestScalar) {
  torch_xla::ir::NodePtr scalar = torch_xla::ir::ops::ScalarOp(1.0, xla::F32);
  ASSERT_TRUE(scalar != nullptr);
}
