#include <gtest/gtest.h>

#include "absl/strings/str_cat.h"

#include "torch_xla/csrc/device.h"

static void CheckFormatError(const std::string& spec) {
  absl::StatusOr<torch::lazy::BackendDevice> r =
      torch_xla::SafeParseDeviceString(spec);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status().message(),
            absl::StrCat("expected the device string `", spec,
                         "` to be in the format: `<type>:<index>`."));
}

static void CheckIndexParseError(const std::string& spec) {
  absl::StatusOr<torch::lazy::BackendDevice> r =
      torch_xla::SafeParseDeviceString(spec);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(
      r.status().message(),
      absl::StrCat("error while parsing the device spec `", spec, "`: stoi"));
}

TEST(DeviceTest, ParseDeviceStringFormatError) {
  CheckFormatError("");
  CheckFormatError("xla");
  CheckFormatError("xla:1:other");
}

TEST(DeviceTest, ParseDeviceStringIndexParseError) {
  CheckIndexParseError("xla:");
  CheckIndexParseError("xla:xla");
  CheckIndexParseError("xla:x11");
}
