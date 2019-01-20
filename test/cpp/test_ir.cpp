#include <gtest/gtest.h>

#include "cpp_test_util.h"
#include "ir.h"
#include "lowering_context.h"
#include "ops/ops.h"
#include "ops/scalar.h"

namespace torch_xla {
namespace cpp_test {

TEST(IrTest, TestScalarCreate) {
  ir::NodePtr scalar = ir::ops::ScalarOp(1.0, xla::F32);
  ASSERT_TRUE(scalar != nullptr);
}

}  // namespace cpp_test
}  // namespace torch_xla
