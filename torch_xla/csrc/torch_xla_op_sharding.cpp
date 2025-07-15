#include "torch_xla_op_sharding.h"

namespace torch_xla {

OpSharding::OpSharding() {}

OpSharding::OpSharding(
    const xla::OpSharding& op_sharding,
    const std::optional<std::vector<int64_t>>& denormalized_tile_assignment)
    : op_sharding_(std::make_unique<xla::OpSharding>(op_sharding)),
      denormalized_tile_assignment_(
          denormalized_tile_assignment.value_or(std::vector<int64_t>{})) {}

OpSharding::OpSharding(const OpSharding& other)
    : denormalized_tile_assignment_(other.denormalized_tile_assignment_) {
  if (other.op_sharding_) {
    op_sharding_ = std::make_unique<xla::OpSharding>(*other.op_sharding_);
  } else {
    // Fallback to default replicated sharding
    op_sharding_ = std::make_unique<xla::OpSharding>();
    op_sharding_->set_type(xla::OpSharding::REPLICATED);
  }
}

OpSharding& OpSharding::operator=(const OpSharding& other) {
  if (this != &other) {
    if (other.op_sharding_) {
      op_sharding_ = std::make_unique<xla::OpSharding>(*other.op_sharding_);
    } else {
      // Fallback to default replicated sharding
      op_sharding_ = std::make_unique<xla::OpSharding>();
      op_sharding_->set_type(xla::OpSharding::REPLICATED);
    }
    denormalized_tile_assignment_ = other.denormalized_tile_assignment_;
  }
  return *this;
}

OpSharding::OpSharding(OpSharding&& other) noexcept
    : op_sharding_(std::move(other.op_sharding_)),
      denormalized_tile_assignment_(
          std::move(other.denormalized_tile_assignment_)) {
  // other.op_sharding_ is now nullptr, which is safe
}

OpSharding& OpSharding::operator=(OpSharding&& other) noexcept {
  if (this != &other) {
    op_sharding_ = std::move(other.op_sharding_);
    denormalized_tile_assignment_ =
        std::move(other.denormalized_tile_assignment_);
  }
  return *this;
}

// Forwarded methods from xla::OpSharding for API compatibility
xla::OpSharding::Type OpSharding::type() const { return op_sharding_->type(); }

bool OpSharding::replicate_on_last_tile_dim() const {
  return op_sharding_->replicate_on_last_tile_dim();
}

int OpSharding::tile_assignment_dimensions_size() const {
  return op_sharding_->tile_assignment_dimensions_size();
}

int OpSharding::tile_assignment_devices_size() const {
  return op_sharding_->tile_assignment_devices_size();
}

int OpSharding::tile_assignment_dimensions(int index) const {
  return op_sharding_->tile_assignment_dimensions(index);
}

int OpSharding::tile_assignment_devices(int index) const {
  return op_sharding_->tile_assignment_devices(index);
}

std::string OpSharding::DebugString() const {
  return op_sharding_->DebugString();
}

const ::google::protobuf::RepeatedField<int64_t>&
OpSharding::iota_reshape_dims() const {
  return op_sharding_->iota_reshape_dims();
}

const ::google::protobuf::RepeatedField<int64_t>&
OpSharding::tile_assignment_dimensions() const {
  return op_sharding_->tile_assignment_dimensions();
}

const ::google::protobuf::RepeatedField<int64_t>&
OpSharding::tile_assignment_devices() const {
  return op_sharding_->tile_assignment_devices();
}

const ::google::protobuf::RepeatedField<int32_t>&
OpSharding::iota_transpose_perm() const {
  return op_sharding_->iota_transpose_perm();
}

const ::google::protobuf::RepeatedField<int32_t>& OpSharding::last_tile_dims()
    const {
  return op_sharding_->last_tile_dims();
}

const xla::ShapeProto& OpSharding::tile_shape() const {
  return op_sharding_->tile_shape();
}

const xla::OpSharding& OpSharding::GetXlaOpSharding() const {
  return *op_sharding_;
}

xla::OpSharding& OpSharding::GetMutableXlaOpSharding() { return *op_sharding_; }

const std::vector<int64_t>& OpSharding::GetDenormalizedTileAssignment() const {
  return denormalized_tile_assignment_;
}

}  // namespace torch_xla
