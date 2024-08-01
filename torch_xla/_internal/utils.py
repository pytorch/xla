import re


def parse_xla_device(device: str):
  m = re.match(r'([A-Z]+):(\d+)$', device)
  if m:
    return (m.group(1), int(m.group(2)))
