#include "test/cpp/test_status_common.h"

using torch_xla::StatusTest;

namespace torch_xla::cpp_test {
namespace {

// This file instantiates the parameterized tests defined in
// `test_status_common.h`. It specifically configures the test environment by
// explicitly setting the `TORCH_SHOW_CPP_STACKTRACES` environment variable to
// 'true' in the test fixture's `SetUp` method.
//
// Any new `TEST_P` test cases added to `test_status_common.h` will
// automatically be run in this mode (with C++ error context).
INSTANTIATE_WITH_CPP_STACKTRACES_MODE(StatusWithCppErrorContextTest, StatusTest,
                                      kShow);

}  // namespace
}  // namespace torch_xla::cpp_test
