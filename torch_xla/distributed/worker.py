from __future__ import division
from __future__ import print_function

class Worker(object):

  def __init__(self, internal_ip, machine_type, zone):
    if not isinstance(internal_ip, str):
      raise ValueError('internal_ip must be of type str')
    self._internal_ip = internal_ip
    if not isinstance(machine_type, str):
      raise ValueError('machine_type must be of type str')
    self._machine_type = machine_type
    if not isinstance(zone, str):
      raise ValueError('zone must be of type str')
    self._zone = zone


class ClientWorker(Worker):

  def __init__(self, internal_ip, machine_type, zone, hostname=None):
    super(ClientWorker, self).__init__(internal_ip, machine_type, zone)
    if hostname is not None and not isinstance(hostname, str):
      raise ValueError('hostname must be of type str')
    self._hostname = hostname

  def __repr__(self):
    return ('{{{internal_ip}, {machine_type}, {zone},'
            ' {hostname}}}').format(
                internal_ip=self._internal_ip,
                machine_type=self._machine_type,
                zone=self._zone,
                hostname=self._hostname)

  def __eq__(self, other):
    return (self._internal_ip == other._internal_ip and
            self._machine_type == other._machine_type and
            self._zone == other._zone and self._hostname == other._hostname)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(repr(self))

  def __repr__(self):
    return ('{{{internal_ip}, {machine_type}, {zone},'
            ' {hostname}}}').format(
                internal_ip=self._internal_ip,
                machine_type=self._machine_type,
                zone=self._zone,
                hostname=self._hostname)

  def __eq__(self, other):
    return (self._internal_ip == other._internal_ip and
            self._machine_type == other._machine_type and
            self._zone == other._zone and self._hostname == other._hostname)

  def __ne__(self, other):
    return not self.__eq__(self, other)

  def __hash__(self):
    return hash(repr(self))


class ServiceWorker(Worker):

  def __init__(self, internal_ip, port, machine_type, zone, runtime_version):
    super(ServiceWorker, self).__init__(internal_ip, machine_type, zone)
    self._port = int(port)
    if not isinstance(runtime_version, str):
      raise ValueError('runtime_version must be of type str')
    self._runtime_version = runtime_version

  def __repr__(self):
    return ('{{{internal_ip}, {port}, {machine_type}, {zone},'
            ' {runtime_version}}}').format(
                internal_ip=self._internal_ip,
                port=self._port,
                machine_type=self._machine_type,
                zone=self._zone,
                runtime_version=self._runtime_version)

  def __eq__(self, other):
    return (self._internal_ip == other._internal_ip and
            self._port == other._port and
            self._machine_type == other._machine_type and
            self._zone == other._zone and
            self._runtime_version == other._runtime_version)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(repr(self))
