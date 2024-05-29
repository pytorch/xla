(torch310) root@23587be5ee07:/pytorch/xla# PJRT_DEVICE=TPU python test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py WhileLoopTest.test_while_loop_tpu_MNIST_inside_loop_with_mutation_in_batchnorm2d
print and check behavior by exporting the model
class GraphModule(torch.nn.Module):
    def forward(self, iteri, x, y):
        iteri: "i64[]"; x: "f32[16, 1, 28, 28]"; y: "f32[16, 10]"; 
    
        iteri, x, y, = fx_pytree.tree_flatten_spec(([iteri, x, y], {}), self._in_spec)
        # No stacktrace found for following nodes
        conv1_weight: "f32[10, 1, 5, 5]" = self.conv1.weight
        conv1_bias: "f32[10]" = self.conv1.bias
        conv2_weight: "f32[20, 10, 5, 5]" = self.conv2.weight
        conv2_bias: "f32[20]" = self.conv2.bias
        fc1_weight: "f32[50, 500]" = self.fc1.weight
        fc1_bias: "f32[50]" = self.fc1.bias
        fc2_weight: "f32[10, 50]" = self.fc2.weight
        fc2_bias: "f32[10]" = self.fc2.bias
        
        # File: /pytorch/torch/_higher_order_ops/while_loop.py:124 in while_loop, code: return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
        while_loop_cond_graph_0 = self.while_loop_cond_graph_0
        while_loop_body_graph_0 = self.while_loop_body_graph_0
        while_loop = torch.ops.higher_order.while_loop(while_loop_cond_graph_0, while_loop_body_graph_0, (iteri, x, y), (conv1_bias, conv1_weight, conv2_bias, conv2_weight, fc1_bias, fc1_weight, fc2_bias, fc2_weight));  while_loop_cond_graph_0 = while_loop_body_graph_0 = iteri = x = y = conv1_bias = conv1_weight = conv2_bias = conv2_weight = fc1_bias = fc1_weight = fc2_bias = fc2_weight = None
        getitem: "i64[]" = while_loop[0]
        getitem_1: "f32[16, 1, 28, 28]" = while_loop[1]
        getitem_2: "f32[16, 10]" = while_loop[2];  while_loop = None
        return pytree.tree_unflatten((getitem, getitem_1, getitem_2), self._out_spec)
        
    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[16, 1, 28, 28]", arg2_1: "f32[16, 10]", arg3_1: "f32[10]", arg4_1: "f32[10, 1, 5, 5]", arg5_1: "f32[20]", arg6_1: "f32[20, 10, 5, 5]", arg7_1: "f32[50]", arg8_1: "f32[50, 500]", arg9_1: "f32[10]", arg10_1: "f32[10, 50]"):
            # File: /pytorch/torch/_higher_order_ops/while_loop.py:124 in while_loop, code: return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
            gt: "b8[]" = torch.ops.aten.gt.Scalar(arg0_1, 0);  arg0_1 = None
            return gt
            
    class <lambda>(torch.nn.Module):
        def forward(self, arg0_1: "i64[]", arg1_1: "f32[16, 1, 28, 28]", arg2_1: "f32[16, 10]", arg3_1: "f32[10]", arg4_1: "f32[10, 1, 5, 5]", arg5_1: "f32[20]", arg6_1: "f32[20, 10, 5, 5]", arg7_1: "f32[50]", arg8_1: "f32[50, 500]", arg9_1: "f32[10]", arg10_1: "f32[10, 50]"):
            # File: /pytorch/torch/_higher_order_ops/while_loop.py:124 in while_loop, code: return while_loop_op(cond_fn, body_fn, carried_inputs, additional_inputs)
            conv2d: "f32[16, 10, 28, 28]" = torch.ops.aten.conv2d.default(arg1_1, arg4_1, arg3_1, [1, 1], [2, 2]);  arg4_1 = arg3_1 = None
            max_pool2d: "f32[16, 10, 14, 14]" = torch.ops.aten.max_pool2d.default(conv2d, [2, 2]);  conv2d = None
            relu: "f32[16, 10, 14, 14]" = torch.ops.aten.relu.default(max_pool2d);  max_pool2d = None
            empty: "u8[0]" = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='xla', index=0))
            _native_batch_norm_legit = torch.ops.aten._native_batch_norm_legit.no_stats(relu, None, None, True, 0.1, 1e-05);  relu = None
            getitem: "f32[16, 10, 14, 14]" = _native_batch_norm_legit[0]
            getitem_1: "f32[10]" = _native_batch_norm_legit[1]
            getitem_2: "f32[10]" = _native_batch_norm_legit[2];  _native_batch_norm_legit = None
            conv2d_1: "f32[16, 20, 10, 10]" = torch.ops.aten.conv2d.default(getitem, arg6_1, arg5_1);  getitem = arg6_1 = arg5_1 = None
            max_pool2d_1: "f32[16, 20, 5, 5]" = torch.ops.aten.max_pool2d.default(conv2d_1, [2, 2]);  conv2d_1 = None
            relu_1: "f32[16, 20, 5, 5]" = torch.ops.aten.relu.default(max_pool2d_1);  max_pool2d_1 = None
            empty_1: "u8[0]" = torch.ops.aten.empty.memory_format([0], dtype = torch.uint8, layout = torch.strided, device = device(type='xla', index=0))
            _native_batch_norm_legit_1 = torch.ops.aten._native_batch_norm_legit.no_stats(relu_1, None, None, True, 0.1, 1e-05);  relu_1 = None
            getitem_3: "f32[16, 20, 5, 5]" = _native_batch_norm_legit_1[0]
            getitem_4: "f32[20]" = _native_batch_norm_legit_1[1]
            getitem_5: "f32[20]" = _native_batch_norm_legit_1[2];  _native_batch_norm_legit_1 = None
            view: "f32[16, 500]" = torch.ops.aten.view.default(getitem_3, [16, 500]);  getitem_3 = None
            linear: "f32[16, 50]" = torch.ops.aten.linear.default(view, arg8_1, arg7_1);  view = arg8_1 = arg7_1 = None
            relu_2: "f32[16, 50]" = torch.ops.aten.relu.default(linear);  linear = None
            linear_1: "f32[16, 10]" = torch.ops.aten.linear.default(relu_2, arg10_1, arg9_1);  relu_2 = arg10_1 = arg9_1 = None
            sub: "i64[]" = torch.ops.aten.sub.Tensor(arg0_1, 1);  arg0_1 = None
            clone: "f32[16, 1, 28, 28]" = torch.ops.aten.clone.default(arg1_1);  arg1_1 = None
            log_softmax: "f32[16, 10]" = torch.ops.aten.log_softmax.int(linear_1, 1);  linear_1 = None
            return (sub, clone, log_softmax)
            
after print and check behavior by exporting the model
additional_inputs:  0  size:  torch.Size([10])
additional_inputs:  1  size:  torch.Size([10, 1, 5, 5])
additional_inputs:  2  size:  torch.Size([20])
additional_inputs:  3  size:  torch.Size([20, 10, 5, 5])
additional_inputs:  4  size:  torch.Size([50])
additional_inputs:  5  size:  torch.Size([50, 500])
additional_inputs:  6  size:  torch.Size([10])
additional_inputs:  7  size:  torch.Size([10, 50])
arrive 2 !!!
body_result:  0  size:  torch.Size([])
body_result:  1  size:  torch.Size([10])
body_result:  2  size:  torch.Size([10, 1, 5, 5])
body_result:  3  size:  torch.Size([20])
body_result:  4  size:  torch.Size([20, 10, 5, 5])
body_result:  5  size:  torch.Size([50])
body_result:  6  size:  torch.Size([50, 500])
body_result:  7  size:  torch.Size([10])
body_result:  8  size:  torch.Size([10, 50])
body_result:  9  size:  torch.Size([16, 1, 28, 28])
body_result:  10  size:  torch.Size([16, 10])
res:  FunctionalTensor(lvl=0, value=\
tensor([[-2.6107, -2.0759, -2.3293, -2.1662, -2.2920, -2.3025, -2.3766, -1.9861,
         -2.5690, -2.5095],
        [-2.1087, -2.3078, -2.5448, -2.2745, -2.4400, -2.2127, -2.1260, -2.0672,
         -2.5276, -2.5833],
        [-2.2926, -2.4044, -2.3817, -2.1380, -2.1055, -2.5277, -2.2644, -2.2526,
         -2.4681, -2.2731],
        [-2.2662, -2.4601, -2.6416, -2.1145, -2.1916, -2.1509, -2.3809, -2.0078,
         -2.4798, -2.5221],
        [-2.4818, -2.3978, -2.4498, -2.0514, -2.3918, -2.3066, -2.0950, -2.2117,
         -2.3430, -2.3976],
        [-2.3397, -2.1403, -2.5003, -2.2138, -2.0800, -2.4003, -2.3998, -2.4113,
         -2.2665, -2.3544],
        [-2.1328, -2.0456, -2.6789, -2.3585, -2.2183, -2.3677, -2.6154, -2.1635,
         -2.1672, -2.4777],
        [-2.2627, -2.1659, -2.4703, -2.2919, -2.1487, -2.4828, -2.3760, -2.4754,
         -2.0619, -2.3940],
        [-2.3335, -2.1922, -2.4700, -2.3684, -2.3040, -2.3420, -2.5318, -2.0962,
         -2.1356, -2.3377],
        [-2.4228, -2.6091, -2.3183, -2.2272, -2.1279, -2.2599, -2.5270, -1.9462,
         -2.3669, -2.3912],
        [-2.4600, -2.0824, -2.2538, -2.5079, -2.3923, -2.0765, -2.5656, -2.1534,
         -2.3229, -2.3459],
        [-2.2736, -2.5169, -2.2825, -2.3244, -2.2392, -2.3194, -2.2582, -2.0248,
         -2.4000, -2.4729],
        [-2.2319, -2.2860, -2.6737, -2.5501, -2.1282, -2.2258, -2.1155, -2.2400,
         -2.5053, -2.2238],
        [-2.3137, -2.3190, -2.4314, -2.2866, -2.1347, -2.5438, -2.5545, -1.9783,
         -2.3948, -2.2152],
        [-2.6169, -2.2272, -2.2467, -2.3641, -2.0573, -2.4324, -2.4008, -2.3753,
         -2.1191, -2.3028],
        [-2.4255, -2.2824, -2.2752, -2.3070, -2.4135, -2.1329, -2.3059, -2.0833,
         -2.4603, -2.4123]], device='xla:0'))
expected_res:  tensor([[-2.6107, -2.0759, -2.3293, -2.1662, -2.2920, -2.3025, -2.3766, -1.9861,
         -2.5690, -2.5095],
        [-2.1087, -2.3078, -2.5448, -2.2745, -2.4400, -2.2127, -2.1260, -2.0672,
         -2.5276, -2.5833],
        [-2.2926, -2.4044, -2.3817, -2.1380, -2.1055, -2.5277, -2.2644, -2.2526,
         -2.4681, -2.2731],
        [-2.2662, -2.4601, -2.6416, -2.1145, -2.1916, -2.1509, -2.3809, -2.0078,
         -2.4798, -2.5221],
        [-2.4818, -2.3978, -2.4498, -2.0514, -2.3918, -2.3066, -2.0950, -2.2117,
         -2.3430, -2.3976],
        [-2.3397, -2.1403, -2.5003, -2.2138, -2.0800, -2.4003, -2.3998, -2.4113,
         -2.2665, -2.3544],
        [-2.1328, -2.0456, -2.6789, -2.3585, -2.2183, -2.3677, -2.6154, -2.1635,
         -2.1672, -2.4777],
        [-2.2627, -2.1659, -2.4703, -2.2919, -2.1487, -2.4828, -2.3760, -2.4754,
         -2.0619, -2.3940],
        [-2.3335, -2.1922, -2.4700, -2.3684, -2.3040, -2.3420, -2.5318, -2.0962,
         -2.1356, -2.3377],
        [-2.4228, -2.6091, -2.3183, -2.2272, -2.1279, -2.2599, -2.5270, -1.9462,
         -2.3669, -2.3912],
        [-2.4600, -2.0824, -2.2538, -2.5079, -2.3923, -2.0765, -2.5656, -2.1534,
         -2.3229, -2.3459],
        [-2.2736, -2.5169, -2.2825, -2.3244, -2.2392, -2.3194, -2.2582, -2.0248,
         -2.4000, -2.4729],
        [-2.2319, -2.2860, -2.6737, -2.5501, -2.1282, -2.2258, -2.1155, -2.2400,
         -2.5053, -2.2238],
        [-2.3137, -2.3190, -2.4314, -2.2866, -2.1347, -2.5438, -2.5545, -1.9783,
         -2.3948, -2.2152],
        [-2.6169, -2.2272, -2.2467, -2.3641, -2.0573, -2.4324, -2.4008, -2.3753,
         -2.1191, -2.3028],
        [-2.4255, -2.2824, -2.2752, -2.3070, -2.4135, -2.1329, -2.3059, -2.0833,
         -2.4603, -2.4123]], device='xla:0')
.
----------------------------------------------------------------------
Ran 1 test in 5.756s

OK
(torch310) root@23587be5ee07:/pytorch/xla# 
