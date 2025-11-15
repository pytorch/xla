#include <gtest/gtest.h>
#include <torch/csrc/lazy/backend/backend_device.h>

#include <cstdint>
#include <string>
#include <string_view>

#include "absl/strings/str_cat.h"

#include "torch_xla/csrc/device.h"

namespace torch_xla {

static void CheckFormatError(const std::string& spec) {
  absl::StatusOr<torch::lazy::BackendDevice> r = SafeParseDeviceString(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.status().message(),
            absl::StrCat("expected the device string `", spec,
                         "` to be in the format: `<type>:<index>`."));
}

static void CheckIndexParseError(const std::string& spec) {
  absl::StatusOr<torch::lazy::BackendDevice> r = SafeParseDeviceString(spec);
  ASSERT_FALSE(r.ok());
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

static void CheckDeviceTypeConstructionWithString(
    XlaDeviceType xla_device_type, std::string_view device_type_str) {
  DeviceType device_type(device_type_str);
  EXPECT_EQ(device_type.getType(), xla_device_type);
  EXPECT_EQ(device_type.toString(), absl::StrCat(device_type_str, ":"));
}

TEST(DeviceTest, ConstructNativeDeviceTypeWithString) {
#define XLA_NATIVE_DEVICE_TYPE_CHECK_CONSTRUCTION_WITH_STRING(type, _) \
  CheckDeviceTypeConstructionWithString(XlaDeviceType::type, #type);
  XLA_FOR_ALL_NATIVE_DEVICE_TYPES_(
      XLA_NATIVE_DEVICE_TYPE_CHECK_CONSTRUCTION_WITH_STRING)
#undef XLA_DEVICE_TYPE_CHECK_CONSTRUCTION_WITH_STRING
}

TEST(DeviceTest, ConstructPluginDeviceTypeWithString) {
  DeviceType device_type("OTHER");
  EXPECT_EQ(device_type.getType(), XlaDeviceType::PLUGIN);
  EXPECT_EQ(device_type.toString(), "OTHER:");
}

static void CheckDeviceTypeConstructionWithEnum(
    XlaDeviceType xla_device_type, std::string_view device_type_str) {
  DeviceType device_type(xla_device_type);
  ASSERT_EQ(device_type.getType(), xla_device_type);
  EXPECT_EQ(device_type.toString(), absl::StrCat(device_type_str, ":"));
}

TEST(DeviceTest, ConstructNativeDeviceTypeWithEnum) {
#define XLA_NATIVE_DEVICE_TYPE_CHECK_CONSTRUCTION_WITH_ENUM(type, _) \
  CheckDeviceTypeConstructionWithEnum(XlaDeviceType::type, #type);
  XLA_FOR_ALL_NATIVE_DEVICE_TYPES_(
      XLA_NATIVE_DEVICE_TYPE_CHECK_CONSTRUCTION_WITH_ENUM)
#undef XLA_DEVICE_TYPE_CHECK_CONSTRUCTION_WITH_STRING
}

TEST(DeviceTest, ConstructPluginDeviceTypeWithEnumError) {
  EXPECT_DEATH({ DeviceType device_type(XlaDeviceType::PLUGIN); },
               absl::StrCat("invalid native XlaDeviceType value: ",
                            static_cast<int8_t>(XlaDeviceType::PLUGIN)));
}

}  // namespace torch_xla
