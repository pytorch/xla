# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
from collections.abc import MutableMapping

import tomlkit
from tomlkit.items import Array

project_name = 'torch_xla'

_current_torch_version = '2.6.0'
# The following should be updated after each new torch release.
_latest_torch_version_on_pypi = '2.6.0'

_libtpu_version = '0.0.8'

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  assert spec is not None
  module = importlib.util.module_from_spec(spec)
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(project_name)
__version__ = _version_module._get_version_for_build()
_jax_version = _version_module._version  # JAX version, with no .dev suffix.
_cmdclass = _version_module._get_cmdclass(project_name)
_minimum_torch_version = _version_module._minimum_torch_version

with open('pyproject.toml', 'r') as f:
  data = tomlkit.load(f)

project = data['project']
assert isinstance(project, MutableMapping)
dependencies = project['dependencies']
assert isinstance(dependencies, Array)
od = project['optional-dependencies']
assert isinstance(od, MutableMapping)

assert isinstance(dependencies[0], str)
assert dependencies[0].startswith('torch')
dependencies[0] = f'torch >={_minimum_torch_version}, <={_jax_version}'
od['minimum-torch'] = [f'torch=={_minimum_torch_version}']
od['ci'] = [f'torch=={_latest_torch_version_on_pypi}']
od['tpu'] = [
        f'torch>={_current_torch_version},<={_jax_version}',
        f'libtpu=={_libtpu_version}',
        'requests',  # necessary for jax.distributed.initialize
        ]
od['cuda'] = [
        f"torch=={_current_torch_version}",
        f"jax-cuda12-plugin[with_cuda]>={_current_torch_version},<={_jax_version}",
        ]

od['cuda12'] = [
        f"torch=={_current_torch_version}",
        f"jax-cuda12-plugin[with_cuda]>={_current_torch_version},<={_jax_version}",
        ]

od['cuda12_pip'] = [
        f"torch=={_current_torch_version}",
        f"jax-cuda12-plugin[with_cuda]>={_current_torch_version},<={_jax_version}",
        ]

od['cuda12_local'] = [
        f"torch=={_current_torch_version}",
        f"jax-cuda12-plugin=={_current_torch_version}",
        ]

# ROCm support for ROCm 6.0 and above.
od['rocm'] = [
        f"torch=={_current_torch_version}",
        f"jax-rocm60-plugin>={_current_torch_version},<={_jax_version}",
        ]

with open('pyproject.toml', 'w') as f:
  tomlkit.dump(data, f)
