#!/usr/bin/env python3
"""Tool to distribute training on Cloud TPU Pods."""
from __future__ import division
from __future__ import print_function

import abc

class BaseWorker(abc.ABC):

  @classmethod
  @abc.abstractmethod
  def validate_cluster(cls, workers):
    """Validates the current worker cluster configuration.

    Raises:
      RuntimeError: If the cluster is misconfigured, this validation will
      raise a error.
    """
    raise NotImplementedError()


class CpuWorker(BaseWorker):

  def __init__(self,
               internal_ip,
               machine_type,
               zone,
               hostname=None):
    self._internal_ip = internal_ip
    self._machine_type = machine_type
    self._zone = zone
    self._hostname = hostname

  @classmethod
  def validate_cluster(cls, workers):
    """Validates the CPU workers configuration.

    Args:
      workers: a list of CpuWorker objects.

    Raises:
      RuntimeError: If the CPU cluster is misconfigured. For example, if the VMs
        are in different zones, or not all of the CPU workers have the same size
        (number of CPU cores, RAM size) we raise an exception as this would
        negatively and significantly affect performance.
    """
    raise NotImplementedError()


class TpuWorker(BaseWorker):

  def __init__(self,
               internal_ip,
               accelerator_type,
               zone):
    self._internal_ip = internal_ip
    self._accelerator_type = accelerator_type
    self._zone = zone

  @classmethod
  def validate_cluster(cls, workers):
    """Validates the TPU worker configuration.

    Args:
      workers: a list of TpuWorker objects.

    Raises:
      RuntimeError: If the TPU cluster is misconfigured. For example, if the
        TPUs are in different zones, or the TPUs are different versions we
        raise an exception as this silently impacts performance.
    """
    raise NotImplementedError()


class ClusterResolver(object):
  """Cluster Resolver for Client VM and Cloud TPU mesh."""

  def __init__(self,
               tpus,
               vms=None,
               zone=None,
               project=None):
    """Creates a new ClusterResolver object."""

    assert tpus, "TPU name list must not be empty."

    self._tpus = tpus
    self._vms = vms
    self._zone = zone
    self._project = project

  def vm_cluster(self):
    """Gets client VM cluster info.

    The instance group that the current VM belongs to is picked up from
    the GCE instance metadata set of the VM. If a list of VMs was used for
    initializing cluster resolver, we use that instead.

    Returns:
      A list of internal IP addresses of client side/red cluster.

    Raises:
      RuntimeError: If the red VM cluster is not healthy or the red VM was
        not created as part of an instance group. If not using an instance
        group, users should just manually configure.
    """
    raise NotImplementedError()

  def tpu_cluster(self):
    """Gets TPU VM cluster info.

    Calls the TPU CLH to get TPU node data and returns list of TPU worker
    VMs internal IP addresses. If zone and project are not specified at
    ClusterResolver init time, we infer these bits from GCE metadata.

    Returns:
      A list of internal IP addresses of TPU cluster.

    Raises:
      RuntimeError: If the TPU DNE or the TPU is in not in HEALTHY state. In the
        sea of donuts case, also raises if they're not all in the same zone.
    """
    raise NotImplementedError()

