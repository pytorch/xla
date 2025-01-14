import torch
import torch_xla
import torch_xla.core.xla_model as xm


"""
   t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
    cpu_inp, cpu_args, cpu_kwargs = cpu(sample_input)

    actual = torch_fn(t_inp, *t_args, **t_kwargs)
    expected = torch_fn(cpu_inp, *cpu_args, **cpu_kwargs)

[test_reference_eager] sample_input: SampleInput(input=7.358110427856445, args=(0,), kwargs={}, broadcasts_input=False, name='')
[test_reference_eager] sample_input: SampleInput(input=7, args=(0,), kwargs={}, broadcasts_input=False, name='')
"""

a = torch.tensor(7.358110427856445, device=xm.xla_device())
print(f'[xla]: {torch.cumsum(a, 0)}')

b = torch.tensor(7.358110427856445)
print(f'[cpu]: {torch.cumsum(b, 0)}')

# import torch_xla.debug.metrics as met

# # For short report that only contains a few key metrics.
# print(met.short_metrics_report())
# # For full report that includes all metrics.
# print(met.metrics_report())