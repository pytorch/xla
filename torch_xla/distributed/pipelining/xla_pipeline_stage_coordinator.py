# Copyright (c) 2024, PyTorch XLA Contributors
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
XLA Pipeline Stage Coordinator for PyTorch Pipeline Parallelism

This module provides XLA-specific pipeline stage coordination for pipeline parallelism,
inheriting from the base PipelineStageCoordinator and overriding only the methods that
need XLA-specific behavior.
"""

import logging
import torch
import torch.distributed as dist
from torch.distributed.pipelining.pipeline_stage_coordinator import PipelineStageCoordinator

# Import XLA modules
import torch_xla
import torch_xla.runtime as xr

logger = logging.getLogger(__name__)


class XlaPipelineStageCoordinator(PipelineStageCoordinator):
  """
    XLA-specific pipeline stage coordinator for pipeline parallelism.
    
    Inherits from the base PipelineStageCoordinator and overrides only the methods
    that need XLA-specific behavior.
    """

  def __init__(self, device: torch.device, group: dist.ProcessGroup):
    # check XLA device and group
    assert device.type == "xla", f"XlaPipelineStageCoordinator: device {device} is not XLA device"
    if group:
      assert isinstance(
          group, dist.ProcessGroup
      ), f"XlaPipelineStageCoordinator: group {group} is not a ProcessGroup"

    # For XLA, we need to ensure the device is set to "cpu" and group is "gloo"
    # regardless of the input device and group params
    device = torch.device("cpu")
    backend = "gloo"
    ranks = list(range(xr._WORLD_SIZE))
    group = dist.new_group(backend=backend, ranks=ranks)
    super().__init__(device, group)
    logger.debug(
        "XLAPipelineStageCoordinator: Initialized with device=%s and group=%s",
        self._device, self._group.name())

  def create_stage_communication_buffer(self, metadata: torch.Tensor,
                                        device: torch.device) -> torch.Tensor:
    """
        Create a tensor buffer from metadata for pipeline stage communication.
        
        XLA-specific implementation: create empty tensor on XLA device.
        
        Args:
            metadata: The metadata object received from another stage
            device: Target XLA device
            
        Returns:
            Empty XLA tensor ready for pipeline stage communication
        """
    logger.debug(
        "XlaPipelineStageCoordinator: Creating stage communication buffer from metadata - shape %s, dtype %s on device %s",
        metadata.shape, metadata.dtype, device)

    # For XLA, we need to ensure the device is set to "xla"
    # regardless of the input device parameter
    return torch.empty(metadata.shape, dtype=metadata.dtype, device="xla")


def register_xla_pipeline_stage_coordinator():
  """Register the XLA pipeline stage coordinator with PyTorch's registry."""
  logger.debug("Attempting to register XLA pipeline stage coordinator")
  try:
    # Import the registration function from PyTorch
    from torch.distributed.pipelining.pipeline_stage_coordinator import register_pipeline_stage_coordinator

    # Register XLA coordinator
    def create_xla_coordinator(device, group):
      logger.debug("Creating XLA pipeline stage coordinator instance")
      return XlaPipelineStageCoordinator(device, group)

    register_pipeline_stage_coordinator(
        torch.device("xla"), create_xla_coordinator)
    logger.debug("Successfully registered XLA pipeline stage coordinator")

  except ImportError as e:
    logger.debug(
        "Failed to register XLA pipeline stage coordinator due to import error: %s",
        e)
    # If PyTorch doesn't have the pipeline stage coordinator infrastructure yet,
    # this will be a no-op. The coordinator can still be used directly.
    pass
