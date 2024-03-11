"""This module contains implementation of ATen ops."""
import torch
import jax
import jax.numpy as jnp
from torch_xla2.ops import op_base

# Keys are OpOverload, value is a callable that takes
# XLATensor2
all_ops = {}


all_ops[torch.ops.aten.add.Tensor] = op_base.BinaryOpWithPromotion(jnp.add)
all_ops[torch.ops.aten.sub.Tensor] = op_base.BinaryOpWithPromotion(jnp.subtract)
all_ops[torch.ops.aten.sub.Scalar] = op_base.BinaryOpWithPromotion(jnp.subtract)
all_ops[torch.ops.aten.mul.Tensor] = op_base.BinaryOpWithPromotion(jnp.multiply)
all_ops[torch.ops.aten.div.Tensor] = op_base.BinaryOpWithPromotion(jnp.divide)
