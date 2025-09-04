#ifndef XLA_TORCH_XLA_CSRC_TORCH_XLA_OP_SHARDING_H_
#define XLA_TORCH_XLA_CSRC_TORCH_XLA_OP_SHARDING_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "google/protobuf/repeated_field.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"

namespace torch_xla {

// Wrapper class for xla::OpSharding that provides additional functionality
// and maintains denormalized tile assignment information.
//
// This class serves as a bridge between PyTorch XLA's sharding representation
// and XLA's native OpSharding, allowing for extended functionality while
// maintaining compatibility with the underlying XLA infrastructure.
class OpSharding {
 public:
  // Default constructor
  OpSharding();

  // Constructs OpSharding from xla::OpSharding with optional denormalized
  // tile assignment
  explicit OpSharding(const xla::OpSharding& op_sharding,
                      const std::optional<std::vector<int64_t>>&
                          denormalized_tile_assignment = std::nullopt);

  // Copy constructor
  OpSharding(const OpSharding& other);

  // Copy assignment operator
  OpSharding& operator=(const OpSharding& other);

  // Move constructor
  OpSharding(OpSharding&& other) noexcept;

  // Move assignment operator
  OpSharding& operator=(OpSharding&& other) noexcept;

  // Destructor (default is sufficient due to unique_ptr)
  ~OpSharding() = default;

  // Forwarded methods from xla::OpSharding for API compatibility
  xla::OpSharding::Type type() const;
  bool replicate_on_last_tile_dim() const;
  int tile_assignment_dimensions_size() const;
  int tile_assignment_devices_size() const;
  int tile_assignment_dimensions(int index) const;
  int tile_assignment_devices(int index) const;
  std::string DebugString() const;
  const ::google::protobuf::RepeatedField<int64_t>& iota_reshape_dims() const;
  const ::google::protobuf::RepeatedField<int64_t>& tile_assignment_dimensions()
      const;
  const ::google::protobuf::RepeatedField<int64_t>& tile_assignment_devices()
      const;
  const ::google::protobuf::RepeatedField<int32_t>& iota_transpose_perm() const;
  const ::google::protobuf::RepeatedField<int32_t>& last_tile_dims() const;
  const xla::ShapeProto& tile_shape() const;

  // Access to underlying xla::OpSharding
  const xla::OpSharding& GetXlaOpSharding() const;
  xla::OpSharding& GetMutableXlaOpSharding();

  // Access to denormalized tile assignment
  const std::vector<int64_t>& GetDenormalizedTileAssignment() const;

 private:
  // Underlying XLA OpSharding object
  std::unique_ptr<xla::OpSharding> op_sharding_;

  // Additional denormalized tile assignment information
  std::vector<int64_t> denormalized_tile_assignment_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_TORCH_XLA_OP_SHARDING_H_
