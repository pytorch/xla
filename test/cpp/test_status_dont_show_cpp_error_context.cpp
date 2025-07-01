#include <gtest/gtest.h>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/status.h"

namespace torch_xla {

TEST(StatusWithoutErrorContextTest, MaybeWithLocationRetunsSameStatus) {
  absl::Status error_status = absl::InvalidArgumentError("Test error message");
  absl::Status result = MaybeWithLocation(error_status, "test_file.cpp", 42);
  EXPECT_EQ(result, error_status);
}

TEST(StatusWithoutErrorContextTest, MaybeWithNewMessageEmptyNewMessage) {
  absl::Status error_status = absl::InvalidArgumentError("Original error");
  absl::Status result = MaybeWithNewMessage(error_status, "test_file.cpp", 42);
  EXPECT_EQ(result, error_status);
}

TEST(StatusWithoutErrorContextTest, MaybeWithNewMessageNonEmptyNewMessage) {
  constexpr char new_err_string[] = "New error message";
  absl::Status error_status = absl::InvalidArgumentError("Original error");
  absl::Status result =
      MaybeWithNewMessage(error_status, "test_file.cpp", 42, new_err_string);

  ASSERT_FALSE(result.ok());
  ASSERT_NE(result, error_status);
  EXPECT_EQ(result.code(), error_status.code());
  EXPECT_EQ(result.message(), new_err_string);
}

TEST(StatusWithoutErrorContextTest, MacroReturnIfErrorWithError) {
  constexpr char err_string[] = "Test error";

  auto test_function = [=]() -> absl::Status {
    absl::Status error_status = absl::InvalidArgumentError(err_string);
    XLA_RETURN_IF_ERROR(error_status);
    return absl::OkStatus();
  };

  absl::Status result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(result.message(), err_string);
}

TEST(StatusWithoutErrorContextTest, MacroAssignOrReturnWithError) {
  auto test_function = []() -> absl::StatusOr<int> {
    absl::StatusOr<int> status_or = absl::InvalidArgumentError("Test error");
    XLA_ASSIGN_OR_RETURN(int value, status_or);
    return value * 2;
  };

  absl::StatusOr<int> result = test_function();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(StatusWithoutErrorContextTest, MacroErrorWithLocation) {
  absl::Status error_status = absl::InvalidArgumentError("Test error");
  absl::Status result = XLA_ERROR_WITH_LOCATION(error_status);
  EXPECT_EQ(result, error_status);
}

void SetUp() {
  setenv(runtime::env::kEnvShowCppErrorContext, /* value= */ "false",
         /* replace= */ 1);
}

}  // namespace torch_xla

int main(int argc, char **argv) {
  ::torch_xla::SetUp();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
