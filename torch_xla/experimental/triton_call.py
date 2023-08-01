import xla_client as xc

def triton_call(*args: Union[jax.Array, bool, int, float],
    kernel: triton.JITFunction,
    out_shape: Union[ShapeDtype, Sequence[ShapeDtype]],
    grid: GridOrLambda,
    call_name: str = "triton_kernel_call",
    num_warps: int = 4,
    num_stages: int = 2,
    input_output_aliases: Optional[Dict[int, int]] = None,
    zeroed_outputs: Union[
        Sequence[int], Callable[[Dict[str, Any]], Sequence[int]]
    ] = (),
    debug: bool = False,
    serialized_metadata: bytes = b"",
    **metaparams: Any,
):

    # Run output shape function over the inputs to get the output shape

    
  )
