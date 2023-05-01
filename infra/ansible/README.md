# Ansible playbook

This ansible playbook will perform the following actions on the localhost:
  * install required pip and apt packages, depending on the specified stage,
    architecture and accelerator (see [apt.yaml](config/apt.yaml) and
    [pip.yaml](config/pip.yaml)).
  * fetch bazel (version configured in [vars.yaml](config/vars.yaml)),
  * fetch PyTorch and XLA sources at master (or specific revisions,
    see role `fetch_srcs` in [playbook.yaml](playbook.yaml)).
  * set required environment variables (see [env.yaml](config/env.yaml)),
  * build and install PyTorch and XLA wheels,
  * apply infrastructure tests (see `*/tests.yaml` files in [roles](roles)).

## Prerequisites

* Python 3.8+
* Ansible. Install with `pip install ansible`.

## Running

The playbook requires passing explicitly 3 variables that configure playbook
behavior (installed pip/apt packages and set environment variables):
* `stage`: build or release. Different packages are installed depending on
  the chosen stage.
* `arch`: aarch64 or amd64. Architecture of the built image and wheels.
* `accelerator`: tpu or cuda. Available accelerator.

The variables can be passed through `-e` flag: `-e "<var>=<value>"`.

Example: `ansible-playbook playbook.yaml -e "stage=build arch=amd64 accelerator=tpu"`

## Config structure

The playbook configuration is split into 4 files, per each logical system.
The configuration is simply loaded as playbook variables which are then passed
to specific roles and tasks.
Only variables in [config/env.yaml](config/env.yaml) are passed as env variables.

* [apt.yaml](config/apt.yaml) - specifies apt packages for each stage and
  architecture or accelerator.
  Packages shared between all architectures and accelerators in a given stage
  are specified in `*_common`. They are appended to any architecture specific list.

  This config also contains a list of required apt repos and signing keys.
  These variables are mainly consumed by the [install_deps](roles/install_deps/tasks/main.yaml) role.

* [pip.yaml](config/pip.yaml) - similarly to apt.yaml, lists pip packages per stage and arch / accelerator.
  In both pip and apt config files stage and and arch / accelerator are
  concatenated together and specified under one key (e.g. build_amd64, release_tpu).

* [env.yaml](config/env.yaml) - contains Ansible variables that are passed as env variables when
  building PyTorch and XLA (`build_env`). Variables in `release_env` are saved in `/etc/environment` (executed for the `release` stage).

* [vars.yaml](config/vars.yaml) - Ansible variables used in other config files and throughout the playbook.
  Not associated with any particular system.

Variables from these config files are dynamically loaded (during playbook execution),
see [playbook.yaml](playbook.yaml).
