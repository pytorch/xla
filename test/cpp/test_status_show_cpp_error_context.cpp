#include "test/cpp/test_status_common.h"

// This file instantiates the parameterized tests defined in
// `test_status_common.h`. It specifically configures the test environment by
// explicitly setting the `XLA_SHOW_CPP_ERROR_CONTEXT` environment variable to
// 'true' in the test fixture's `SetUp` method.
//
// Any new `TEST_P` test cases added to `test_status_common.h` will
// automatically be run in this mode (with C++ error context).
INSTANTIATE_TEST_SUITE_WITH_MODE(SHOW);
