(torch310) root@23587be5ee07:/pytorch/xla# PJRT_DEVICE=TPU python test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py WhileLoopTest.test_while_loop_tpu_MNIST_inside_loop_with_mutation_in_batchnorm2d
print and check behavior by exporting the model
class GraphModule(torch.nn.Module):
    def forward(self, iteri, x, y):
        iteri: "i64[]"; x: "f32[16, 1, 28, 28]"; y: "f32[16, 10]"; 
    
        iteri, x, y, = fx_pytree.tree_flatten_spec(([iteri, x, y], {}), self._in_spec)
        # No stacktrace found for following nodes
        conv1_weight: "f32[10, 1, 5, 5]" = self.conv1.weight
        conv1_bias: "f32[10]" = self.conv1.bias
        bn1_weight: "f32[10]" = self.bn1.weight
        bn1_bias: "f32[10]" = self.bn1.bias
        conv2_weight: "f32[20, 10, 5, 5]" = self.conv2.weight
        conv2_bias: "f32[20]" = self.conv2.bias
        bn2_weight: "f32[20]" = self.bn2.weight
        bn2_bias: "f32[20]" = self.bn2.bias
        fc1_weight: "f32[50, 500]" = self.fc1.weight
        fc1_bias: "f32[50]" = self.fc1.bias
        fc2_weight: "f32[10, 50]" = self.fc2.weight
        fc2_bias: "f32[10]" = self.fc2.bias
        bn1_num_batches_tracked: "i64[]" = self.bn1.num_batches_tracked
        bn1_running_mean: "f32[10]" = self.bn1.running_mean
        bn1_running_var: "f32[10]" = self.bn1.running_var
        bn2_num_batches_tracked: "i64[]" = self.bn2.num_batches_tracked
        bn2_running_mean: "f32[20]" = self.bn2.running_mean
        bn2_running_var: "f32[20]" = self.bn2.running_var
        
        # File: /pytorch/torch/_higher_order_ops/while_loop.py:124 in while_loop, code: return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (iteri, x, y), (bn1_bias, bn1_num_batches_tracked, bn1_running_mean, bn1_running_var, bn1_weight, bn2_bias, bn2_num_batches_tracked, bn2_running_mean, bn2_running_var, bn2_weight, conv1_bias, conv1_weight, conv2_bias, conv2_weight, fc1_bias, fc1_weight, fc2_bias, fc2_weight));  while_loop_cond_graph_0 = while_loop_body_graph_0 = iteri = x = y = bn1_bias = bn1_num_batches_tracked = bn1_running_mean = bn1_running_var = bn1_weight = bn2_bias = bn2_num_batches_tracked = bn2_running_mean = bn2_running_var = bn2_weight = conv1_bias = conv1_weight = conv2_bias = conv2_weight = fc1_bias = fc1_weight = fc2_bias = fc2_weight = None
        getitem: "i64[]" = while_loop[0]
        getitem_1: "f32[16, 1, 28, 28]" = while_loop[1]
        getitem_2: "f32[16, 10]" = while_loop[2];  while_loop = None
        return pytree.tree_unflatten((getitem, getitem_1, getitem_2), self._out_spec)
        
    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[16, 1, 28, 28]", arg2_1: "f32[16, 10]", arg3_1: "f32[10]", arg4_1: "i64[]", arg5_1: "f32[10]", arg6_1: "f32[10]", arg7_1: "f32[10]", arg8_1: "f32[20]", arg9_1: "i64[]", arg10_1: "f32[20]", arg11_1: "f32[20]", arg12_1: "f32[20]", arg13_1: "f32[10]", arg14_1: "f32[10, 1, 5, 5]", arg15_1: "f32[20]", arg16_1: "f32[20, 10, 5, 5]", arg17_1: "f32[50]", arg18_1: "f32[50, 500]", arg19_1: "f32[10]", arg20_1: "f32[10, 50]"):
            # File: /pytorch/torch/_higher_order_ops/while_loop.py:124 in while_loop, code: return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
            gt: "b8[]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
            return gt
            
    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[16, 1, 28, 28]", arg2_1: "f32[16, 10]", arg3_1: "f32[10]", arg4_1: "i64[]", arg5_1: "f32[10]", arg6_1: "f32[10]", arg7_1: "f32[10]", arg8_1: "f32[20]", arg9_1: "i64[]", arg10_1: "f32[20]", arg11_1: "f32[20]", arg12_1: "f32[20]", arg13_1: "f32[10]", arg14_1: "f32[10, 1, 5, 5]", arg15_1: "f32[20]", arg16_1: "f32[20, 10, 5, 5]", arg17_1: "f32[50]", arg18_1: "f32[50, 500]", arg19_1: "f32[10]", arg20_1: "f32[10, 50]"):
            # File: /pytorch/torch/_higher_order_ops/while_loop.py:124 in while_loop, code: return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
            conv2d: "f32[16, 10, 28, 28]" = torch.ops.aten.conv2d.default(arg1_1, arg14_1, arg13_1, [1, 1], [2, 2]);  arg14_1 = arg13_1 = None
            max_pool2d: "f32[16, 10, 14, 14]" = torch.ops.aten.max_pool2d.default(conv2d, [2, 2]);  conv2d = None
            relu: "f32[16, 10, 14, 14]" = torch.ops.aten.relu.default(max_pool2d);  max_pool2d = None
            add: "i64[]" = torch.ops.aten.add.Tensor(arg4_1, 1)
            _propagate_xla_data = torch.ops.aten._propagate_xla_data.default(arg4_1, add);  arg4_1 = add = None
            empty: "u8[0]" = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='xla', index=0))
            _native_batch_norm_legit_functional = torch.ops.aten._native_batch_norm_legit_functional.default(relu, arg7_1, arg3_1, arg5_1, arg6_1, True, 0.1, 1e-05);  relu = arg7_1 = arg3_1 = None
            getitem: "f32[16, 10, 14, 14]" = _native_batch_norm_legit_functional[0]
            getitem_1: "f32[10]" = _native_batch_norm_legit_functional[1]
            getitem_2: "f32[10]" = _native_batch_norm_legit_functional[2]
            getitem_3: "f32[10]" = _native_batch_norm_legit_functional[3]
            getitem_4: "f32[10]" = _native_batch_norm_legit_functional[4];  _native_batch_norm_legit_functional = None
            _propagate_xla_data_1 = torch.ops.aten._propagate_xla_data.default(arg5_1, getitem_3);  arg5_1 = getitem_3 = None
            _propagate_xla_data_2 = torch.ops.aten._propagate_xla_data.default(arg6_1, getitem_4);  arg6_1 = getitem_4 = None
            conv2d_1: "f32[16, 20, 10, 10]" = torch.ops.aten.conv2d.default(getitem, arg16_1, arg15_1);  getitem = arg16_1 = arg15_1 = None
            max_pool2d_1: "f32[16, 20, 5, 5]" = torch.ops.aten.max_pool2d.default(conv2d_1, [2, 2]);  conv2d_1 = None
            relu_1: "f32[16, 20, 5, 5]" = torch.ops.aten.relu.default(max_pool2d_1);  max_pool2d_1 = None
            add_1: "i64[]" = torch.ops.aten.add.Tensor(arg9_1, 1)
            _propagate_xla_data_3 = torch.ops.aten._propagate_xla_data.default(arg9_1, add_1);  arg9_1 = add_1 = None
            empty_1: "u8[0]" = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='xla', index=0))
            _native_batch_norm_legit_functional_1 = torch.ops.aten._native_batch_norm_legit_functional.default(relu_1, arg12_1, arg8_1, arg10_1, arg11_1, True, 0.1, 1e-05);  relu_1 = arg12_1 = arg8_1 = None
            getitem_5: "f32[16, 20, 5, 5]" = _native_batch_norm_legit_functional_1[0]
            getitem_6: "f32[20]" = _native_batch_norm_legit_functional_1[1]
            getitem_7: "f32[20]" = _native_batch_norm_legit_functional_1[2]
            getitem_8: "f32[20]" = _native_batch_norm_legit_functional_1[3]
            getitem_9: "f32[20]" = _native_batch_norm_legit_functional_1[4];  _native_batch_norm_legit_functional_1 = None
            _propagate_xla_data_4 = torch.ops.aten._propagate_xla_data.default(arg10_1, getitem_8);  arg10_1 = getitem_8 = None
            _propagate_xla_data_5 = torch.ops.aten._propagate_xla_data.default(arg11_1, getitem_9);  arg11_1 = getitem_9 = None
            view: "f32[16, 500]" = torch.ops.aten.view.default(getitem_5, [16, 500]);  getitem_5 = None
            linear: "f32[16, 50]" = torch.ops.aten.linear.default(view, arg18_1, arg17_1);  view = arg18_1 = arg17_1 = None
            relu_2: "f32[16, 50]" = torch.ops.aten.relu.default(linear);  linear = None
            linear_1: "f32[16, 10]" = torch.ops.aten.linear.default(relu_2, arg20_1, arg19_1);  relu_2 = arg20_1 = arg19_1 = None
            sub: "i64[]" = torch.ops.aten.sub.Tensor(arg0_1, 1);  arg0_1 = None
            clone: "f32[16, 1, 28, 28]" = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
            log_softmax: "f32[16, 10]" = torch.ops.aten.log_softmax.int(linear_1, 1);  linear_1 = None
            return (sub, clone, log_softmax)
            
after print and check behavior by exporting the model
E
======================================================================
ERROR: test_while_loop_tpu_MNIST_inside_loop_with_mutation_in_batchnorm2d (__main__.WhileLoopTest)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/pytorch/xla/test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py", line 489, in test_while_loop_tpu_MNIST_inside_loop_with_mutation_in_batchnorm2d
    _, _, res = mnist(iteri, l_in_0, l_out)
  File "/pytorch/torch/nn/modules/module.py", line 1532, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/pytorch/torch/nn/modules/module.py", line 1541, in _call_impl
    return forward_call(*args, **kwargs)
  File "/pytorch/xla/test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py", line 465, in forward
    return while_loop(cond_fn, body_fn, (iteri, x, y))
  File "/pytorch/torch/_higher_order_ops/while_loop.py", line 141, in while_loop
    return torch.compile(while_loop_op, backend="eager", fullgraph=True)(
  File "/pytorch/torch/_dynamo/eval_frame.py", line 421, in _fn
    return fn(*args, **kwargs)
  File "/pytorch/torch/_dynamo/external_utils.py", line 35, in inner
    @functools.wraps(fn)
  File "/pytorch/torch/_dynamo/eval_frame.py", line 560, in _fn
    return fn(*args, **kwargs)
  File "<eval_with_key>.33", line 28, in forward
    while_loop = torch.ops.higher_order.while_loop(cond_fn_0, body_fn_0, (l_args_2_0_, l_args_2_1_, l_args_2_2_), (l__args___1___closure___0_cell_contents_bn1_bias, l__args___1___closure___0_cell_contents_bn1_num_batches_tracked, l__args___1___closure___0_cell_contents_bn1_running_mean, l__args___1___closure___0_cell_contents_bn1_running_var, l__args___1___closure___0_cell_contents_bn1_weight, l__args___1___closure___0_cell_contents_bn2_bias, l__args___1___closure___0_cell_contents_bn2_num_batches_tracked, l__args___1___closure___0_cell_contents_bn2_running_mean, l__args___1___closure___0_cell_contents_bn2_running_var, l__args___1___closure___0_cell_contents_bn2_weight, l__args___1___closure___0_cell_contents_conv1_bias, l__args___1___closure___0_cell_contents_conv1_weight, l__args___1___closure___0_cell_contents_conv2_bias, l__args___1___closure___0_cell_contents_conv2_weight, l__args___1___closure___0_cell_contents_fc1_bias, l__args___1___closure___0_cell_contents_fc1_weight, l__args___1___closure___0_cell_contents_fc2_bias, l__args___1___closure___0_cell_contents_fc2_weight));  cond_fn_0 = body_fn_0 = l_args_2_0_ = l_args_2_1_ = l_args_2_2_ = l__args___1___closure___0_cell_contents_bn1_bias = l__args___1___closure___0_cell_contents_bn1_num_batches_tracked = l__args___1___closure___0_cell_contents_bn1_running_mean = l__args___1___closure___0_cell_contents_bn1_running_var = l__args___1___closure___0_cell_contents_bn1_weight = l__args___1___closure___0_cell_contents_bn2_bias = l__args___1___closure___0_cell_contents_bn2_num_batches_tracked = l__args___1___closure___0_cell_contents_bn2_running_mean = l__args___1___closure___0_cell_contents_bn2_running_var = l__args___1___closure___0_cell_contents_bn2_weight = l__args___1___closure___0_cell_contents_conv1_bias = l__args___1___closure___0_cell_contents_conv1_weight = l__args___1___closure___0_cell_contents_conv2_bias = l__args___1___closure___0_cell_contents_conv2_weight = l__args___1___closure___0_cell_contents_fc1_bias = l__args___1___closure___0_cell_contents_fc1_weight = l__args___1___closure___0_cell_contents_fc2_bias = l__args___1___closure___0_cell_contents_fc2_weight = None
  File "/pytorch/torch/_higher_order_ops/while_loop.py", line 56, in __call__
    return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)
  File "/pytorch/torch/_ops.py", line 379, in __call__
    return wrapper()
  File "/pytorch/torch/_dynamo/eval_frame.py", line 560, in _fn
    return fn(*args, **kwargs)
  File "/pytorch/torch/_ops.py", line 375, in wrapper
    return self.dispatch(
  File "/pytorch/torch/_ops.py", line 296, in dispatch
    return kernel(*args, **kwargs)
  File "/pytorch/torch/_higher_order_ops/utils.py", line 63, in inner
    return autograd_not_implemented_inner(op, deferred_error, *args, **kwargs)
  File "/pytorch/torch/_higher_order_ops/utils.py", line 36, in autograd_not_implemented_inner
    result = operator(*args, **kwargs)
  File "/pytorch/torch/_higher_order_ops/while_loop.py", line 56, in __call__
    return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)
  File "/pytorch/torch/_ops.py", line 379, in __call__
    return wrapper()
  File "/pytorch/torch/_dynamo/eval_frame.py", line 560, in _fn
    return fn(*args, **kwargs)
  File "/pytorch/torch/_ops.py", line 375, in wrapper
    return self.dispatch(
  File "/pytorch/torch/_ops.py", line 358, in dispatch
    return kernel(*args, **kwargs)
  File "/pytorch/torch/_ops.py", line 149, in functionalize_dk_fn
    return fn(_CppFunctionalizeAPI(), *args, **kwargs)
  File "/pytorch/torch/_higher_order_ops/while_loop.py", line 253, in while_loop_func
    raise UnsupportedAliasMutationException(
torch._higher_order_ops.utils.UnsupportedAliasMutationException: torch.while_loop's body_fn might be modifying the input!

----------------------------------------------------------------------
Ran 1 test in 4.577s

FAILED (errors=1)
(torch310) root@23587be5ee07:/pytorch/xla# 
