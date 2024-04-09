"""This module contains implementation of ATen ops."""

import torch
import jax
import jax.numpy as jnp
from torch_xla2.ops import op_base

# Keys are OpOverload, value is a callable that takes
# XLATensor2
all_ops = {}
