#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {

TEST(StatusTest, MaybeThrowWithOkStatus) {
  absl::Status ok_status = absl::OkStatus();
  EXPECT_NO_THROW(MaybeThrow(ok_status));
}

TEST(StatusTest, MaybeThrowWithErrorStatus) {
  absl::Status error_status = absl::InvalidArgumentError("Test error");
  EXPECT_THROW(MaybeThrow(error_status), std::runtime_error);
}

TEST(StatusTest, GetValueOrThrowWithOkStatusOr) {
  int value = 42;
  absl::StatusOr<int> status_or = value;
  int result = GetValueOrThrow(std::move(status_or));
  EXPECT_EQ(result, value);
}

TEST(StatusTest, GetValueOrThrowWithErrorStatusOr) {
  absl::StatusOr<int> status_or = absl::InvalidArgumentError("Test error");
  EXPECT_THROW(GetValueOrThrow(std::move(status_or)), std::runtime_error);
}

TEST(StatusTest, MacroReturnIfError) {
  int value = 42;

  auto test_function = [=]() -> absl::StatusOr<int> {
    absl::Status ok_status = absl::OkStatus();
    XLA_RETURN_IF_ERROR(ok_status);
    return value;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), value);
}

TEST(StatusTest, MacroAssignOrReturn) {
  int initial_value = 42;
  int expected_value = initial_value * 2;

  auto test_function = [=]() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = initial_value;
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.value(), expected_value);
}

}  // namespace torch_xla
