I'm using XLA to run exactly one flash attention pallas kernel layer. I have one version using for loop and another version using the scan higher order operator.. Both loops over (trivially) that single layer. I implement the scan myself. But it looks like the result is wrong. There are many repeated numbers. I think the attention HLOs got broken by the scan. I've attached the output HLO for for loop and scan. Can you dive deep to compare them and find the bug?

Debug numerical error from HLO


```
WARNING:root:libtpu.so and TPU device found. Setting PJRT_DEVICE=TPU.

############### Begin for loop ###############




HloModule IrToHlo.72, entry_computation_layout={(s64[], f32[], f32[], f32[2,4,256,256]{3,2,1,0})->(f32[2,4,256,256]{3,2,1,0})}

ENTRY %IrToHlo.72 (p0.2: s64[], p1.11: f32[], p2.12: f32[], p3.54: f32[2,4,256,256]) -> (f32[2,4,256,256]) {
  %constant.9 = s64[] constant(2531011)
  %constant.7 = s64[] constant(214013)
  %constant.5 = s64[] constant(2531011)
  %constant.3 = s64[] constant(214013)
  %p0.2 = s64[] parameter(0), sharding={replicated}
  %multiply.4 = s64[] multiply(s64[] %constant.3, s64[] %p0.2)
  %add.6 = s64[] add(s64[] %constant.5, s64[] %multiply.4)
  %multiply.8 = s64[] multiply(s64[] %constant.7, s64[] %add.6)
  %add.10 = s64[] add(s64[] %constant.9, s64[] %multiply.8)
  %convert.13 = u64[] convert(s64[] %add.10)
  %reshape.15 = u64[1]{0} reshape(u64[] %convert.13)
  %constant.14 = u64[] constant(0)
  %reshape.16 = u64[1]{0} reshape(u64[] %constant.14)
  %concatenate.17 = u64[2]{0} concatenate(u64[1]{0} %reshape.15, u64[1]{0} %reshape.16), dimensions={0}
  %rng-bit-generator.18 = (u64[2]{0}, u32[256]{0}) rng-bit-generator(u64[2]{0} %concatenate.17), algorithm=rng_default
  %get-tuple-element.20 = u64[2]{0} get-tuple-element((u64[2]{0}, u32[256]{0}) %rng-bit-generator.18), index=0
  %convert.33 = u64[] convert(s64[] %add.6)
  %reshape.35 = u64[1]{0} reshape(u64[] %convert.33)
  %constant.34 = u64[] constant(0)
  %reshape.36 = u64[1]{0} reshape(u64[] %constant.34)
  %concatenate.37 = u64[2]{0} concatenate(u64[1]{0} %reshape.35, u64[1]{0} %reshape.36), dimensions={0}
  %rng-bit-generator.38 = (u64[2]{0}, u32[256,256]{1,0}) rng-bit-generator(u64[2]{0} %concatenate.37), algorithm=rng_default
  %get-tuple-element.40 = u64[2]{0} get-tuple-element((u64[2]{0}, u32[256,256]{1,0}) %rng-bit-generator.38), index=0
  %p3.54 = f32[2,4,256,256]{3,2,1,0} parameter(3), sharding={devices=[4,1,1,1]0,1,2,3}
  %custom-call.57 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[2,4,256,256]{3,2,1,0} %p3.54), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %custom-call.56 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[2,4,256,256]{3,2,1,0} %p3.54), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %custom-call.55 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[2,4,256,256]{3,2,1,0} %p3.54), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %custom-call.58 = (f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) custom-call(f32[1,4,256,256]{3,2,1,0} %custom-call.57, f32[1,4,256,256]{3,2,1,0} %custom-call.56, f32[1,4,256,256]{3,2,1,0} %custom-call.55), custom_call_target="tpu_custom_call", operand_layout_constraints={f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,256]{3,2,1,0}}, backend_config={"custom_call_config": {"body": "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ExcDAzsLAwEVBjsDAQcZBxsDAwELAwEDAwELAwEJBAEJAQMdHwcRAfUHAyM7CQEBAQEBAQEBAwMzMQMBEQczDQMBBQUJAwMTKwMBDwcTDQMBBQsNAwM5MQMBGQc5DQMBBQ8RAwMTKwMBDwcTDQMBBQcVEweBfQMTBRMXAwM7CwMBFQY7AwEHGQcbAwMBCwMBAwMBCwMBCQQBCQEDHR8HEQH3BwMNDwkBAQEBAQEBAQMDAQsDAQMDAQsDAQkEAQkBAwUJBxEB+QcDDQ8JAQEBAQEBAQEDAwELAwEDAwELAwEJBAEJAQMFCQcRAQoCBwMNDwkBAQEBAQEBAQMDAQsDAQMDAQsDAQkEAQkBAwUJBgMBBQEAHhSnESkLGQsZEw0JCR0hJREbLSMdCyMhIyktHwsNFR0dJRsVFQsLfxsZGRkZGRkxDQsRCyWNHSUdEw1jtxcTFxcvExcXIxsXFxcZIxkVJR8PDw0JHRFidWlsdGluAHN0YWJsZV9tb3NhaWMAdHB1AGFyaXRoAHZlY3RvcgBtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLnNoYXBlX2Nhc3QAZnVuYy5mdW5jAGZ1bmMucmV0dXJuAHZlY3Rvci5icm9hZGNhc3QAdmVjdG9yLmxvYWQAYXJpdGgubXVsaQBhcml0aC5hZGRpAGFyaXRoLmNtcGkAYXJpdGguc2VsZWN0AHRwdS52ZWN0b3Jfc3RvcmUAYXJpdGguc3ViaQB0cHUubWF0bXVsAHRwdS5pb3RhAHZlY3Rvci5tdWx0aV9yZWR1Y3Rpb24AYXJpdGguYWRkZgBhcml0aC5zdWJmAG1hdGguZXhwAGFyaXRoLmRpdmYAL3Vzci9sb2NhbC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvZmxhc2hfYXR0ZW50aW9uLnB5AF9mbGFzaF9hdHRlbnRpb25fa2VybmVsX3NpbmdsZV9iYXRjaF9zaW5nbGVfc3RlcAB2YWx1ZQBzeW1fbmFtZQBmdW5jdGlvbl90eXBlAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL3dvcmtzcGFjZXMvdG9yY2gvcHl0b3JjaC94bGEvdG9yY2hfeGxhL2V4cGVyaW1lbnRhbC9jdXN0b21fa2VybmVsLnB5AC9icm9hZGNhc3RfaW5fZGltAC9hZGQAZm9yd2FyZAAvZ2V0AC9zd2FwAF9mbGFzaF9hdHRlbnRpb25fa2VybmVsAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAHRyYW5zZm9ybV8zAHRyYW5zZm9ybV80AHRyYW5zZm9ybV81AGt2X2luZGV4X21hcAAvd29ya3NwYWNlcy90b3JjaC9weXRvcmNoL3hsYS90ZXN0L3Rlc3RfYXNfc3RyaWRlX3VzZV9zbGljZS5weQAvbXVsAC9zdWIAcHJlZGljYXRlAC9zZWxlY3RfbgAvZG90X2dlbmVyYWwAZGltZW5zaW9uX251bWJlcnMAdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGRpbWVuc2lvbgAvaW90YQBraW5kAHJlZHVjdGlvbl9kaW1zAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAGJlbG93X29yX29uX2RpYWcAX2ZsYXNoX2F0dGVudGlvbl9pbXBsAHRyYWNlX3BhbGxhcwB3cmFwcGVyAGZhX2N1c3RvbV9mb3J3YXJkAGZsYXNoX2F0dGVudGlvbgBvdmVyZmxvd0ZsYWdzAC9ndAAvbGUAL3BqaXQAZmFzdG1hdGgAL3JlZHVjZV9tYXgAL2V4cAAvcmVkdWNlX3N1bQAvZGl2AG9wZXJhbmRTZWdtZW50U2l6ZXMAc3RyaWRlcwA=", "cost_estimate": {"flops": 269221888, "transcendentals": 262144, "bytes_accessed": 5242880}, "serialization_format": 1, "needs_layout_passes": true}}
  %get-tuple-element.60 = f32[1,4,256,128]{3,2,1,0} get-tuple-element((f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) %custom-call.58), index=1
  %get-tuple-element.61 = f32[1,4,256,128]{3,2,1,0} get-tuple-element((f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) %custom-call.58), index=2
  %get-tuple-element.59 = f32[1,4,256,256]{3,2,1,0} get-tuple-element((f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) %custom-call.58), index=0
  %custom-call.62 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[1,4,256,256]{3,2,1,0} %get-tuple-element.59), custom_call_target="Sharding", sharding={manual}
  %custom-call.63 = f32[2,4,256,256]{3,2,1,0} custom-call(f32[1,4,256,256]{3,2,1,0} %custom-call.62), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1,1,1]0,1,2,3}
  %reshape.64 = f32[2048,256]{1,0} reshape(f32[2,4,256,256]{3,2,1,0} %custom-call.63)
  %get-tuple-element.39 = u32[256,256]{1,0} get-tuple-element((u64[2]{0}, u32[256,256]{1,0}) %rng-bit-generator.38), index=1
  %constant.41 = u32[] constant(9)
  %broadcast.42 = u32[256,256]{1,0} broadcast(u32[] %constant.41), dimensions={}
  %shift-right-logical.43 = u32[256,256]{1,0} shift-right-logical(u32[256,256]{1,0} %get-tuple-element.39, u32[256,256]{1,0} %broadcast.42)
  %convert.44 = f32[256,256]{1,0} convert(u32[256,256]{1,0} %shift-right-logical.43)
  %constant.45 = f32[] constant(1.1920929e-07)
  %broadcast.46 = f32[256,256]{1,0} broadcast(f32[] %constant.45), dimensions={}
  %multiply.47 = f32[256,256]{1,0} multiply(f32[256,256]{1,0} %convert.44, f32[256,256]{1,0} %broadcast.46)
  %p1.11 = f32[] parameter(1), sharding={replicated}
  %p2.12 = f32[] parameter(2), sharding={replicated}
  %subtract.48 = f32[] subtract(f32[] %p1.11, f32[] %p2.12)
  %broadcast.49 = f32[256,256]{1,0} broadcast(f32[] %subtract.48), dimensions={}
  %multiply.50 = f32[256,256]{1,0} multiply(f32[256,256]{1,0} %multiply.47, f32[256,256]{1,0} %broadcast.49)
  %broadcast.51 = f32[256,256]{1,0} broadcast(f32[] %p2.12), dimensions={}
  %add.52 = f32[256,256]{1,0} add(f32[256,256]{1,0} %multiply.50, f32[256,256]{1,0} %broadcast.51)
  %transpose.53 = f32[256,256]{0,1} transpose(f32[256,256]{1,0} %add.52), dimensions={1,0}
  %dot.65 = f32[2048,256]{1,0} dot(f32[2048,256]{1,0} %reshape.64, f32[256,256]{0,1} %transpose.53), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %reshape.66 = f32[2,4,256,256]{3,2,1,0} reshape(f32[2048,256]{1,0} %dot.65)
  %get-tuple-element.19 = u32[256]{0} get-tuple-element((u64[2]{0}, u32[256]{0}) %rng-bit-generator.18), index=1
  %constant.21 = u32[] constant(9)
  %broadcast.22 = u32[256]{0} broadcast(u32[] %constant.21), dimensions={}
  %shift-right-logical.23 = u32[256]{0} shift-right-logical(u32[256]{0} %get-tuple-element.19, u32[256]{0} %broadcast.22)
  %convert.24 = f32[256]{0} convert(u32[256]{0} %shift-right-logical.23)
  %constant.25 = f32[] constant(1.1920929e-07)
  %broadcast.26 = f32[256]{0} broadcast(f32[] %constant.25), dimensions={}
  %multiply.27 = f32[256]{0} multiply(f32[256]{0} %convert.24, f32[256]{0} %broadcast.26)
  %subtract.28 = f32[] subtract(f32[] %p1.11, f32[] %p2.12)
  %broadcast.29 = f32[256]{0} broadcast(f32[] %subtract.28), dimensions={}
  %multiply.30 = f32[256]{0} multiply(f32[256]{0} %multiply.27, f32[256]{0} %broadcast.29)
  %broadcast.31 = f32[256]{0} broadcast(f32[] %p2.12), dimensions={}
  %add.32 = f32[256]{0} add(f32[256]{0} %multiply.30, f32[256]{0} %broadcast.31)
  %constant.1 = f32[] constant(1)
  %broadcast.67 = f32[256]{0} broadcast(f32[] %constant.1), dimensions={}
  %multiply.68 = f32[256]{0} multiply(f32[256]{0} %add.32, f32[256]{0} %broadcast.67)
  %broadcast.69 = f32[2,4,256,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.68), dimensions={3}
  %add.70 = f32[2,4,256,256]{3,2,1,0} add(f32[2,4,256,256]{3,2,1,0} %reshape.66, f32[2,4,256,256]{3,2,1,0} %broadcast.69)
  ROOT %tuple.71 = (f32[2,4,256,256]{3,2,1,0}) tuple(f32[2,4,256,256]{3,2,1,0} %add.70)
}




############### End for loop ###############




Output with for loop tensor([[[[ 0.1121,  0.6861,  0.5678,  ...,  0.5434,  0.4260, -1.2716],
          [ 0.7692, -0.3851,  0.5060,  ...,  1.3148, -0.5299,  0.2398],
          [ 0.1338,  0.6621,  0.2311,  ...,  1.1048,  0.6880,  0.3466],
          ...,
          [ 0.1175, -1.3415, -0.8453,  ...,  0.1454,  0.7326, -0.6258],
          [ 1.0795, -0.8921, -0.1300,  ...,  0.6893,  0.6278, -0.4858],
          [ 0.5186,  1.1050,  0.0789,  ...,  0.1388,  0.8621, -1.1064]],

         [[ 0.1556, -0.5855, -0.0617,  ...,  0.3294,  0.1312, -0.5449],
          [ 0.4192,  0.0745,  0.5902,  ...,  0.9082, -0.4553,  0.3067],
          [-0.8679,  0.5193, -1.1310,  ...,  0.1008,  0.0789,  0.0419],
          ...,
          [ 0.1257, -0.3530, -0.3274,  ...,  0.1874, -0.9451,  0.2377],
          [-0.2119,  0.0239, -0.3678,  ..., -0.9844, -0.7037, -0.6487],
          [-0.1291, -0.8700,  0.0455,  ..., -0.1753,  0.6254,  0.3016]],

         [[ 0.0150, -0.7792,  0.7414,  ...,  0.6869, -0.2445,  0.6276],
          [-0.0352, -0.5303,  0.7493,  ...,  0.4480, -0.6163, -0.0406],
          [-0.1720, -0.8248, -0.1586,  ...,  0.5832, -0.0291, -0.1419],
          ...,
          [ 1.0532, -0.3611,  0.8306,  ...,  0.3352, -1.7829, -0.0668],
          [-1.1738,  0.2782, -0.7867,  ..., -0.2007, -1.0293,  0.6290],
          [-0.1742, -0.0765, -0.0934,  ..., -0.7140, -0.3481, -0.1920]],

         [[-0.8838,  0.2789,  1.1109,  ...,  0.9251, -0.0578,  0.3052],
          [ 0.3878, -0.1937,  0.2581,  ..., -0.6772, -1.1078,  0.5129],
          [ 0.6549,  0.5602, -0.3211,  ...,  1.0761, -0.6905,  0.9525],
          ...,
          [ 0.1793, -0.1728, -0.6183,  ...,  0.0940,  0.2154, -0.2858],
          [ 0.4786, -0.1759, -0.0635,  ..., -0.0231, -0.0247, -0.4456],
          [ 0.2452,  0.0630, -0.4674,  ..., -0.6085, -0.6514, -0.5521]]],


        [[[ 0.4610, -0.5845,  0.3681,  ...,  0.4741, -0.8491,  0.6940],
          [ 0.4725, -0.2513,  0.4111,  ...,  0.2175, -0.1920, -0.2859],
          [-0.4320,  0.3245,  0.8849,  ..., -0.9630,  0.0573,  0.5671],
          ...,
          [-0.0465, -0.2538,  1.1180,  ...,  0.1333,  0.0493,  0.2315],
          [-0.4084, -0.6956,  0.1594,  ...,  0.7081, -0.5239, -1.1626],
          [ 0.8020, -0.1330, -0.4450,  ...,  1.2225,  0.5230,  0.3847]],

         [[ 0.4011, -0.7289,  0.5741,  ..., -0.4568, -0.0864, -0.9447],
          [ 1.0388, -0.7855,  0.3037,  ..., -0.0589,  0.0135,  0.3683],
          [ 0.6263,  0.2997,  0.3288,  ..., -0.3761,  0.3243, -0.3744],
          ...,
          [-0.3347, -1.3243, -0.0565,  ...,  0.4283,  0.1496, -0.4047],
          [ 0.1472,  0.0176,  1.2074,  ...,  0.3786, -0.0803, -0.4379],
          [-0.9115,  1.1951, -0.1302,  ..., -0.3963,  0.3834,  0.4224]],

         [[-0.3230,  0.6030,  0.7626,  ..., -0.9403, -0.2903, -0.8670],
          [ 0.7681, -0.4942,  0.7829,  ..., -1.1500, -0.5232, -0.0823],
          [-0.7941,  0.3000,  0.1098,  ..., -0.1590, -0.0169, -1.3443],
          ...,
          [ 0.3742,  0.0212, -1.1684,  ...,  0.5743, -0.4520, -0.2075],
          [-0.3505,  0.3185,  0.3880,  ..., -0.5080,  0.0332, -0.3726],
          [-0.4050,  0.2457, -0.5968,  ...,  0.1263,  1.1219, -0.1949]],

         [[-0.5382,  0.3970, -0.5320,  ..., -0.6823,  0.5036,  0.0331],
          [-0.6337, -0.7485,  0.0313,  ..., -0.1852, -0.5462,  0.7693],
          [-0.6755, -0.5719,  0.1008,  ...,  0.6379, -0.8329,  1.0762],
          ...,
          [ 0.2815, -0.5872,  0.9467,  ..., -0.0567, -0.7793,  0.3821],
          [-1.2428, -1.3761,  0.0478,  ..., -0.1293, -0.0399, -0.3152],
          [-0.6976, -0.4811, -0.1726,  ...,  0.1209,  0.6792,  0.0834]]]],
       device='xla:0', grad_fn=<AddBackward0>)
.
############### Begin scan ###############




HloModule IrToHlo.297, entry_computation_layout={(f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, s64[], f32[], /*index=5*/f32[], f32[2,4,256,256]{3,2,1,0})->(f32[2,4,256,256]{3,2,1,0})}

%FnComputation.101 (p0.103: f32[256], p1.104: f32[256,256], p2.106: f32[2,4,256,256], p3.108: f32[2,4,256,256], p4.110: f32[2,4,256,256], p5.125: f32[2,4,256,256]) -> (f32[2,4,256,256], f32[2,4,256,256], f32[2,4,256,256], f32[2,4,256,256], f32[2,4,256,256], /*index=5*/f32[2,4,256], f32[2,4,256], f32[256,256], f32[2048,256]) {
  %p4.110 = f32[2,4,256,256]{3,2,1,0} parameter(4), sharding={devices=[4,1,1,1]0,1,2,3}
  %custom-call.111 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[2,4,256,256]{3,2,1,0} %p4.110), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %p3.108 = f32[2,4,256,256]{3,2,1,0} parameter(3), sharding={devices=[4,1,1,1]0,1,2,3}
  %custom-call.109 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[2,4,256,256]{3,2,1,0} %p3.108), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %p2.106 = f32[2,4,256,256]{3,2,1,0} parameter(2), sharding={devices=[4,1,1,1]0,1,2,3}
  %custom-call.107 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[2,4,256,256]{3,2,1,0} %p2.106), custom_call_target="SPMDFullToShardShape", sharding={manual}
  %custom-call.112 = (f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) custom-call(f32[1,4,256,256]{3,2,1,0} %custom-call.111, f32[1,4,256,256]{3,2,1,0} %custom-call.109, f32[1,4,256,256]{3,2,1,0} %custom-call.107), custom_call_target="tpu_custom_call", operand_layout_constraints={f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,256]{3,2,1,0}}, backend_config={"custom_call_config": {"body": "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ExcDAzsLAwEVBjsDAQcZBxsDAwELAwEDAwELAwEJBAEJAQMdHwcRAfUHAyM7CQEBAQEBAQEBAwMzMQMBEQczDQMBBQUJAwMTKwMBDwcTDQMBBQsNAwM5MQMBGQc5DQMBBQ8RAwMTKwMBDwcTDQMBBQcVEweBfQMTBRMXAwM7CwMBFQY7AwEHGQcbAwMBCwMBAwMBCwMBCQQBCQEDHR8HEQH3BwMNDwkBAQEBAQEBAQMDAQsDAQMDAQsDAQkEAQkBAwUJBxEB+QcDDQ8JAQEBAQEBAQEDAwELAwEDAwELAwEJBAEJAQMFCQcRAQoCBwMNDwkBAQEBAQEBAQMDAQsDAQMDAQsDAQkEAQkBAwUJBgMBBQEAHhSnESkLGQsZEw0JCR0hJREbLSMdCyMhIyktHwsNFR0dJRsVFQsLfxsZGRkZGRkxDQsRCyWNHSUdEw1jtxcTFxcvExcXIxsXFxcZIxkVJR8PDw0JHRFidWlsdGluAHN0YWJsZV9tb3NhaWMAdHB1AGFyaXRoAHZlY3RvcgBtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLnNoYXBlX2Nhc3QAZnVuYy5mdW5jAGZ1bmMucmV0dXJuAHZlY3Rvci5icm9hZGNhc3QAdmVjdG9yLmxvYWQAYXJpdGgubXVsaQBhcml0aC5hZGRpAGFyaXRoLmNtcGkAYXJpdGguc2VsZWN0AHRwdS52ZWN0b3Jfc3RvcmUAYXJpdGguc3ViaQB0cHUubWF0bXVsAHRwdS5pb3RhAHZlY3Rvci5tdWx0aV9yZWR1Y3Rpb24AYXJpdGguYWRkZgBhcml0aC5zdWJmAG1hdGguZXhwAGFyaXRoLmRpdmYAL3Vzci9sb2NhbC9saWIvcHl0aG9uMy4xMC9zaXRlLXBhY2thZ2VzL2pheC9leHBlcmltZW50YWwvcGFsbGFzL29wcy90cHUvZmxhc2hfYXR0ZW50aW9uLnB5AF9mbGFzaF9hdHRlbnRpb25fa2VybmVsX3NpbmdsZV9iYXRjaF9zaW5nbGVfc3RlcAB2YWx1ZQBzeW1fbmFtZQBmdW5jdGlvbl90eXBlAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAL3dvcmtzcGFjZXMvdG9yY2gvcHl0b3JjaC94bGEvdG9yY2hfeGxhL2V4cGVyaW1lbnRhbC9jdXN0b21fa2VybmVsLnB5AC9icm9hZGNhc3RfaW5fZGltAC9hZGQAZm9yd2FyZAAvZ2V0AC9zd2FwAF9mbGFzaF9hdHRlbnRpb25fa2VybmVsAHRyYW5zZm9ybV8wAHRyYW5zZm9ybV8xAHRyYW5zZm9ybV8yAHRyYW5zZm9ybV8zAHRyYW5zZm9ybV80AHRyYW5zZm9ybV81AGt2X2luZGV4X21hcAAvd29ya3NwYWNlcy90b3JjaC9weXRvcmNoL3hsYS90ZXN0L3Rlc3RfYXNfc3RyaWRlX3VzZV9zbGljZS5weQAvbXVsAC9zdWIAcHJlZGljYXRlAC9zZWxlY3RfbgAvZG90X2dlbmVyYWwAZGltZW5zaW9uX251bWJlcnMAdHJhbnNwb3NlX2xocwB0cmFuc3Bvc2VfcmhzAGRpbWVuc2lvbgAvaW90YQBraW5kAHJlZHVjdGlvbl9kaW1zAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAGJlbG93X29yX29uX2RpYWcAX2ZsYXNoX2F0dGVudGlvbl9pbXBsAHRyYWNlX3BhbGxhcwB3cmFwcGVyAGZhX2N1c3RvbV9mb3J3YXJkAGZsYXNoX2F0dGVudGlvbgBvdmVyZmxvd0ZsYWdzAC9ndAAvbGUAL3BqaXQAZmFzdG1hdGgAL3JlZHVjZV9tYXgAL2V4cAAvcmVkdWNlX3N1bQAvZGl2AG9wZXJhbmRTZWdtZW50U2l6ZXMAc3RyaWRlcwA=", "cost_estimate": {"flops": 269221888, "transcendentals": 262144, "bytes_accessed": 5242880}, "serialization_format": 1, "needs_layout_passes": true}}
  %get-tuple-element.113 = f32[1,4,256,256]{3,2,1,0} get-tuple-element((f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) %custom-call.112), index=0
  %custom-call.116 = f32[1,4,256,256]{3,2,1,0} custom-call(f32[1,4,256,256]{3,2,1,0} %get-tuple-element.113), custom_call_target="Sharding", sharding={manual}
  %custom-call.117 = f32[2,4,256,256]{3,2,1,0} custom-call(f32[1,4,256,256]{3,2,1,0} %custom-call.116), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1,1,1]0,1,2,3}
  %reshape.118 = f32[2048,256]{1,0} reshape(f32[2,4,256,256]{3,2,1,0} %custom-call.117)
  %p1.104 = f32[256,256]{1,0} parameter(1), sharding={replicated}
  %transpose.105 = f32[256,256]{0,1} transpose(f32[256,256]{1,0} %p1.104), dimensions={1,0}
  %dot.119 = f32[2048,256]{1,0} dot(f32[2048,256]{1,0} %reshape.118, f32[256,256]{0,1} %transpose.105), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %reshape.120 = f32[2,4,256,256]{3,2,1,0} reshape(f32[2048,256]{1,0} %dot.119)
  %p0.103 = f32[256]{0} parameter(0), sharding={replicated}
  %constant.102 = f32[] constant(1)
  %broadcast.121 = f32[256]{0} broadcast(f32[] %constant.102), dimensions={}
  %multiply.122 = f32[256]{0} multiply(f32[256]{0} %p0.103, f32[256]{0} %broadcast.121)
  %broadcast.123 = f32[2,4,256,256]{3,2,1,0} broadcast(f32[256]{0} %multiply.122), dimensions={3}
  %add.124 = f32[2,4,256,256]{3,2,1,0} add(f32[2,4,256,256]{3,2,1,0} %reshape.120, f32[2,4,256,256]{3,2,1,0} %broadcast.123)
  %p5.125 = f32[2,4,256,256]{3,2,1,0} parameter(5), sharding={replicated}
  %get-tuple-element.114 = f32[1,4,256,128]{3,2,1,0} get-tuple-element((f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) %custom-call.112), index=1
  %slice.126 = f32[1,4,256,1]{3,2,1,0} slice(f32[1,4,256,128]{3,2,1,0} %get-tuple-element.114), slice={[0:1], [0:4], [0:256], [0:1]}
  %reshape.127 = f32[1,4,256]{2,1,0} reshape(f32[1,4,256,1]{3,2,1,0} %slice.126)
  %custom-call.128 = f32[1,4,256]{2,1,0} custom-call(f32[1,4,256]{2,1,0} %reshape.127), custom_call_target="Sharding", sharding={manual}
  %custom-call.129 = f32[2,4,256]{2,1,0} custom-call(f32[1,4,256]{2,1,0} %custom-call.128), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1,1]0,1,2,3}
  %get-tuple-element.115 = f32[1,4,256,128]{3,2,1,0} get-tuple-element((f32[1,4,256,256]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}, f32[1,4,256,128]{3,2,1,0}) %custom-call.112), index=2
  %slice.130 = f32[1,4,256,1]{3,2,1,0} slice(f32[1,4,256,128]{3,2,1,0} %get-tuple-element.115), slice={[0:1], [0:4], [0:256], [0:1]}
  %reshape.131 = f32[1,4,256]{2,1,0} reshape(f32[1,4,256,1]{3,2,1,0} %slice.130)
  %custom-call.132 = f32[1,4,256]{2,1,0} custom-call(f32[1,4,256]{2,1,0} %reshape.131), custom_call_target="Sharding", sharding={manual}
  %custom-call.133 = f32[2,4,256]{2,1,0} custom-call(f32[1,4,256]{2,1,0} %custom-call.132), custom_call_target="SPMDShardToFullShape", sharding={devices=[4,1,1]0,1,2,3}
  ROOT %tuple.134 = (f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) tuple(f32[2,4,256,256]{3,2,1,0} %add.124, f32[2,4,256,256]{3,2,1,0} %custom-call.117, f32[2,4,256,256]{3,2,1,0} %p5.125, f32[2,4,256,256]{3,2,1,0} %p5.125, f32[2,4,256,256]{3,2,1,0} %p5.125, /*index=5*/f32[2,4,256]{2,1,0} %custom-call.129, f32[2,4,256]{2,1,0} %custom-call.133, f32[256,256]{0,1} %transpose.105, f32[2048,256]{1,0} %reshape.118)
}

%Body.135 (p0.136: (s64[], f32[2,4,256,256], f32[1,256,256], f32[1,256], f32[1,2,4,256,256], /*index=5*/f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256], f32[1,2,4,256], /*index=10*/f32[1,256,256], f32[1,2048,256], f32[2,4,256,256], f32[2,4,256,256], f32[2,4,256,256])) -> (s64[], f32[2,4,256,256], f32[1,256,256], f32[1,256], f32[1,2,4,256,256], /*index=5*/f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256], f32[1,2,4,256], /*index=10*/f32[1,256,256], f32[1,2048,256], f32[2,4,256,256], f32[2,4,256,256], f32[2,4,256,256]) {
  %p0.136 = (s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) parameter(0)
  %get-tuple-element.137 = s64[] get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=0
  %constant.240 = s64[] constant(1)
  %add.241 = s64[] add(s64[] %get-tuple-element.137, s64[] %constant.240)
  %get-tuple-element.140 = f32[1,256]{1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=3
  %constant.158 = s64[] constant(0)
  %broadcast.159 = s64[] broadcast(s64[] %constant.158), dimensions={}
  %dynamic-slice.160 = f32[1,256]{1,0} dynamic-slice(f32[1,256]{1,0} %get-tuple-element.140, s64[] %get-tuple-element.137, s64[] %broadcast.159), dynamic_slice_sizes={1,256}
  %reshape.161 = f32[256]{0} reshape(f32[1,256]{1,0} %dynamic-slice.160)
  %get-tuple-element.139 = f32[1,256,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=2
  %constant.152 = s64[] constant(0)
  %broadcast.153 = s64[] broadcast(s64[] %constant.152), dimensions={}
  %constant.154 = s64[] constant(0)
  %broadcast.155 = s64[] broadcast(s64[] %constant.154), dimensions={}
  %dynamic-slice.156 = f32[1,256,256]{2,1,0} dynamic-slice(f32[1,256,256]{2,1,0} %get-tuple-element.139, s64[] %get-tuple-element.137, s64[] %broadcast.153, s64[] %broadcast.155), dynamic_slice_sizes={1,256,256}
  %reshape.157 = f32[256,256]{1,0} reshape(f32[1,256,256]{2,1,0} %dynamic-slice.156)
  %get-tuple-element.151 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=14
  %get-tuple-element.150 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=13
  %get-tuple-element.149 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=12
  %get-tuple-element.138 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=1
  %call.162 = (f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) call(f32[256]{0} %reshape.161, f32[256,256]{1,0} %reshape.157, f32[2,4,256,256]{3,2,1,0} %get-tuple-element.151, f32[2,4,256,256]{3,2,1,0} %get-tuple-element.150, f32[2,4,256,256]{3,2,1,0} %get-tuple-element.149, /*index=5*/f32[2,4,256,256]{3,2,1,0} %get-tuple-element.138), to_apply=%FnComputation.101
  %get-tuple-element.163 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=0
  %get-tuple-element.141 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=4
  %get-tuple-element.164 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=1
  %broadcast.165 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[2,4,256,256]{3,2,1,0} %get-tuple-element.164), dimensions={1,2,3,4}
  %constant.166 = s64[] constant(0)
  %broadcast.167 = s64[] broadcast(s64[] %constant.166), dimensions={}
  %constant.168 = s64[] constant(0)
  %broadcast.169 = s64[] broadcast(s64[] %constant.168), dimensions={}
  %constant.170 = s64[] constant(0)
  %broadcast.171 = s64[] broadcast(s64[] %constant.170), dimensions={}
  %constant.172 = s64[] constant(0)
  %broadcast.173 = s64[] broadcast(s64[] %constant.172), dimensions={}
  %dynamic-update-slice.174 = f32[1,2,4,256,256]{4,3,2,1,0} dynamic-update-slice(f32[1,2,4,256,256]{4,3,2,1,0} %get-tuple-element.141, f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.165, s64[] %get-tuple-element.137, s64[] %broadcast.167, s64[] %broadcast.169, /*index=5*/s64[] %broadcast.171, s64[] %broadcast.173)
  %get-tuple-element.142 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=5
  %get-tuple-element.175 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=2
  %broadcast.176 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[2,4,256,256]{3,2,1,0} %get-tuple-element.175), dimensions={1,2,3,4}
  %constant.177 = s64[] constant(0)
  %broadcast.178 = s64[] broadcast(s64[] %constant.177), dimensions={}
  %constant.179 = s64[] constant(0)
  %broadcast.180 = s64[] broadcast(s64[] %constant.179), dimensions={}
  %constant.181 = s64[] constant(0)
  %broadcast.182 = s64[] broadcast(s64[] %constant.181), dimensions={}
  %constant.183 = s64[] constant(0)
  %broadcast.184 = s64[] broadcast(s64[] %constant.183), dimensions={}
  %dynamic-update-slice.185 = f32[1,2,4,256,256]{4,3,2,1,0} dynamic-update-slice(f32[1,2,4,256,256]{4,3,2,1,0} %get-tuple-element.142, f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.176, s64[] %get-tuple-element.137, s64[] %broadcast.178, s64[] %broadcast.180, /*index=5*/s64[] %broadcast.182, s64[] %broadcast.184)
  %get-tuple-element.143 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=6
  %get-tuple-element.186 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=3
  %broadcast.187 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[2,4,256,256]{3,2,1,0} %get-tuple-element.186), dimensions={1,2,3,4}
  %constant.188 = s64[] constant(0)
  %broadcast.189 = s64[] broadcast(s64[] %constant.188), dimensions={}
  %constant.190 = s64[] constant(0)
  %broadcast.191 = s64[] broadcast(s64[] %constant.190), dimensions={}
  %constant.192 = s64[] constant(0)
  %broadcast.193 = s64[] broadcast(s64[] %constant.192), dimensions={}
  %constant.194 = s64[] constant(0)
  %broadcast.195 = s64[] broadcast(s64[] %constant.194), dimensions={}
  %dynamic-update-slice.196 = f32[1,2,4,256,256]{4,3,2,1,0} dynamic-update-slice(f32[1,2,4,256,256]{4,3,2,1,0} %get-tuple-element.143, f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.187, s64[] %get-tuple-element.137, s64[] %broadcast.189, s64[] %broadcast.191, /*index=5*/s64[] %broadcast.193, s64[] %broadcast.195)
  %get-tuple-element.144 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=7
  %get-tuple-element.197 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=4
  %broadcast.198 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[2,4,256,256]{3,2,1,0} %get-tuple-element.197), dimensions={1,2,3,4}
  %constant.199 = s64[] constant(0)
  %broadcast.200 = s64[] broadcast(s64[] %constant.199), dimensions={}
  %constant.201 = s64[] constant(0)
  %broadcast.202 = s64[] broadcast(s64[] %constant.201), dimensions={}
  %constant.203 = s64[] constant(0)
  %broadcast.204 = s64[] broadcast(s64[] %constant.203), dimensions={}
  %constant.205 = s64[] constant(0)
  %broadcast.206 = s64[] broadcast(s64[] %constant.205), dimensions={}
  %dynamic-update-slice.207 = f32[1,2,4,256,256]{4,3,2,1,0} dynamic-update-slice(f32[1,2,4,256,256]{4,3,2,1,0} %get-tuple-element.144, f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.198, s64[] %get-tuple-element.137, s64[] %broadcast.200, s64[] %broadcast.202, /*index=5*/s64[] %broadcast.204, s64[] %broadcast.206)
  %get-tuple-element.145 = f32[1,2,4,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=8
  %get-tuple-element.208 = f32[2,4,256]{2,1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=5
  %broadcast.209 = f32[1,2,4,256]{3,2,1,0} broadcast(f32[2,4,256]{2,1,0} %get-tuple-element.208), dimensions={1,2,3}
  %constant.210 = s64[] constant(0)
  %broadcast.211 = s64[] broadcast(s64[] %constant.210), dimensions={}
  %constant.212 = s64[] constant(0)
  %broadcast.213 = s64[] broadcast(s64[] %constant.212), dimensions={}
  %constant.214 = s64[] constant(0)
  %broadcast.215 = s64[] broadcast(s64[] %constant.214), dimensions={}
  %dynamic-update-slice.216 = f32[1,2,4,256]{3,2,1,0} dynamic-update-slice(f32[1,2,4,256]{3,2,1,0} %get-tuple-element.145, f32[1,2,4,256]{3,2,1,0} %broadcast.209, s64[] %get-tuple-element.137, s64[] %broadcast.211, s64[] %broadcast.213, /*index=5*/s64[] %broadcast.215)
  %get-tuple-element.146 = f32[1,2,4,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=9
  %get-tuple-element.217 = f32[2,4,256]{2,1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=6
  %broadcast.218 = f32[1,2,4,256]{3,2,1,0} broadcast(f32[2,4,256]{2,1,0} %get-tuple-element.217), dimensions={1,2,3}
  %constant.219 = s64[] constant(0)
  %broadcast.220 = s64[] broadcast(s64[] %constant.219), dimensions={}
  %constant.221 = s64[] constant(0)
  %broadcast.222 = s64[] broadcast(s64[] %constant.221), dimensions={}
  %constant.223 = s64[] constant(0)
  %broadcast.224 = s64[] broadcast(s64[] %constant.223), dimensions={}
  %dynamic-update-slice.225 = f32[1,2,4,256]{3,2,1,0} dynamic-update-slice(f32[1,2,4,256]{3,2,1,0} %get-tuple-element.146, f32[1,2,4,256]{3,2,1,0} %broadcast.218, s64[] %get-tuple-element.137, s64[] %broadcast.220, s64[] %broadcast.222, /*index=5*/s64[] %broadcast.224)
  %get-tuple-element.147 = f32[1,256,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=10
  %get-tuple-element.226 = f32[256,256]{0,1} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=7
  %broadcast.227 = f32[1,256,256]{2,1,0} broadcast(f32[256,256]{0,1} %get-tuple-element.226), dimensions={1,2}
  %constant.228 = s64[] constant(0)
  %broadcast.229 = s64[] broadcast(s64[] %constant.228), dimensions={}
  %constant.230 = s64[] constant(0)
  %broadcast.231 = s64[] broadcast(s64[] %constant.230), dimensions={}
  %dynamic-update-slice.232 = f32[1,256,256]{2,1,0} dynamic-update-slice(f32[1,256,256]{2,1,0} %get-tuple-element.147, f32[1,256,256]{2,1,0} %broadcast.227, s64[] %get-tuple-element.137, s64[] %broadcast.229, s64[] %broadcast.231)
  %get-tuple-element.148 = f32[1,2048,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.136), index=11
  %get-tuple-element.233 = f32[2048,256]{1,0} get-tuple-element((f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, /*index=5*/f32[2,4,256]{2,1,0}, f32[2,4,256]{2,1,0}, f32[256,256]{0,1}, f32[2048,256]{1,0}) %call.162), index=8
  %broadcast.234 = f32[1,2048,256]{2,1,0} broadcast(f32[2048,256]{1,0} %get-tuple-element.233), dimensions={1,2}
  %constant.235 = s64[] constant(0)
  %broadcast.236 = s64[] broadcast(s64[] %constant.235), dimensions={}
  %constant.237 = s64[] constant(0)
  %broadcast.238 = s64[] broadcast(s64[] %constant.237), dimensions={}
  %dynamic-update-slice.239 = f32[1,2048,256]{2,1,0} dynamic-update-slice(f32[1,2048,256]{2,1,0} %get-tuple-element.148, f32[1,2048,256]{2,1,0} %broadcast.234, s64[] %get-tuple-element.137, s64[] %broadcast.236, s64[] %broadcast.238)
  ROOT %tuple.242 = (s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) tuple(s64[] %add.241, f32[2,4,256,256]{3,2,1,0} %get-tuple-element.163, f32[1,256,256]{2,1,0} %get-tuple-element.139, f32[1,256]{1,0} %get-tuple-element.140, f32[1,2,4,256,256]{4,3,2,1,0} %dynamic-update-slice.174, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0} %dynamic-update-slice.185, f32[1,2,4,256,256]{4,3,2,1,0} %dynamic-update-slice.196, f32[1,2,4,256,256]{4,3,2,1,0} %dynamic-update-slice.207, f32[1,2,4,256]{3,2,1,0} %dynamic-update-slice.216, f32[1,2,4,256]{3,2,1,0} %dynamic-update-slice.225, /*index=10*/f32[1,256,256]{2,1,0} %dynamic-update-slice.232, f32[1,2048,256]{2,1,0} %dynamic-update-slice.239, f32[2,4,256,256]{3,2,1,0} %get-tuple-element.149, f32[2,4,256,256]{3,2,1,0} %get-tuple-element.150, f32[2,4,256,256]{3,2,1,0} %get-tuple-element.151)
}

%Condition.243 (p0.244: (s64[], f32[2,4,256,256], f32[1,256,256], f32[1,256], f32[1,2,4,256,256], /*index=5*/f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256], f32[1,2,4,256], /*index=10*/f32[1,256,256], f32[1,2048,256], f32[2,4,256,256], f32[2,4,256,256], f32[2,4,256,256])) -> pred[] {
  %p0.244 = (s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) parameter(0)
  %get-tuple-element.246 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=1
  %get-tuple-element.247 = f32[1,256,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=2
  %get-tuple-element.248 = f32[1,256]{1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=3
  %get-tuple-element.249 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=4
  %get-tuple-element.250 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=5
  %get-tuple-element.251 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=6
  %get-tuple-element.252 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=7
  %get-tuple-element.253 = f32[1,2,4,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=8
  %get-tuple-element.254 = f32[1,2,4,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=9
  %get-tuple-element.255 = f32[1,256,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=10
  %get-tuple-element.256 = f32[1,2048,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=11
  %get-tuple-element.257 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=12
  %get-tuple-element.258 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=13
  %get-tuple-element.259 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=14
  %get-tuple-element.245 = s64[] get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %p0.244), index=0
  %constant.260 = s64[] constant(1)
  ROOT %compare.261 = pred[] compare(s64[] %get-tuple-element.245, s64[] %constant.260), direction=LT
}

%scan.262 (p0.263: s64[], p1.264: f32[2,4,256,256], p2.265: f32[1,256,256], p3.266: f32[1,256], p4.267: f32[1,2,4,256,256], p5.268: f32[1,2,4,256,256], p6.269: f32[1,2,4,256,256], p7.270: f32[1,2,4,256,256], p8.271: f32[1,2,4,256], p9.272: f32[1,2,4,256], p10.273: f32[1,256,256], p11.274: f32[1,2048,256], p12.275: f32[2,4,256,256], p13.276: f32[2,4,256,256], p14.277: f32[2,4,256,256]) -> (s64[], f32[2,4,256,256], f32[1,256,256], f32[1,256], f32[1,2,4,256,256], /*index=5*/f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256,256], f32[1,2,4,256], f32[1,2,4,256], /*index=10*/f32[1,256,256], f32[1,2048,256], f32[2,4,256,256], f32[2,4,256,256], f32[2,4,256,256]) {
  %p0.263 = s64[] parameter(0)
  %p1.264 = f32[2,4,256,256]{3,2,1,0} parameter(1)
  %p2.265 = f32[1,256,256]{2,1,0} parameter(2)
  %p3.266 = f32[1,256]{1,0} parameter(3)
  %p4.267 = f32[1,2,4,256,256]{4,3,2,1,0} parameter(4)
  %p5.268 = f32[1,2,4,256,256]{4,3,2,1,0} parameter(5)
  %p6.269 = f32[1,2,4,256,256]{4,3,2,1,0} parameter(6)
  %p7.270 = f32[1,2,4,256,256]{4,3,2,1,0} parameter(7)
  %p8.271 = f32[1,2,4,256]{3,2,1,0} parameter(8)
  %p9.272 = f32[1,2,4,256]{3,2,1,0} parameter(9)
  %p10.273 = f32[1,256,256]{2,1,0} parameter(10)
  %p11.274 = f32[1,2048,256]{2,1,0} parameter(11)
  %p12.275 = f32[2,4,256,256]{3,2,1,0} parameter(12)
  %p13.276 = f32[2,4,256,256]{3,2,1,0} parameter(13)
  %p14.277 = f32[2,4,256,256]{3,2,1,0} parameter(14)
  %tuple.278 = (s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) tuple(s64[] %p0.263, f32[2,4,256,256]{3,2,1,0} %p1.264, f32[1,256,256]{2,1,0} %p2.265, f32[1,256]{1,0} %p3.266, f32[1,2,4,256,256]{4,3,2,1,0} %p4.267, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0} %p5.268, f32[1,2,4,256,256]{4,3,2,1,0} %p6.269, f32[1,2,4,256,256]{4,3,2,1,0} %p7.270, f32[1,2,4,256]{3,2,1,0} %p8.271, f32[1,2,4,256]{3,2,1,0} %p9.272, /*index=10*/f32[1,256,256]{2,1,0} %p10.273, f32[1,2048,256]{2,1,0} %p11.274, f32[2,4,256,256]{3,2,1,0} %p12.275, f32[2,4,256,256]{3,2,1,0} %p13.276, f32[2,4,256,256]{3,2,1,0} %p14.277)
  ROOT %while.279 = (s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) while((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %tuple.278), condition=%Condition.243, body=%Body.135
}

ENTRY %IrToHlo.297 (p0.1: f32[2,4,256,256], p1.2: f32[2,4,256,256], p2.3: f32[2,4,256,256], p3.44: s64[], p4.53: f32[], p5.54: f32[], p6.99: f32[2,4,256,256]) -> (f32[2,4,256,256]) {
  %constant.51 = s64[] constant(2531011)
  %constant.49 = s64[] constant(214013)
  %constant.47 = s64[] constant(2531011)
  %constant.45 = s64[] constant(214013)
  %p3.44 = s64[] parameter(3), sharding={replicated}
  %multiply.46 = s64[] multiply(s64[] %constant.45, s64[] %p3.44)
  %add.48 = s64[] add(s64[] %constant.47, s64[] %multiply.46)
  %multiply.50 = s64[] multiply(s64[] %constant.49, s64[] %add.48)
  %add.52 = s64[] add(s64[] %constant.51, s64[] %multiply.50)
  %convert.55 = u64[] convert(s64[] %add.52)
  %reshape.57 = u64[1]{0} reshape(u64[] %convert.55)
  %constant.56 = u64[] constant(0)
  %reshape.58 = u64[1]{0} reshape(u64[] %constant.56)
  %concatenate.59 = u64[2]{0} concatenate(u64[1]{0} %reshape.57, u64[1]{0} %reshape.58), dimensions={0}
  %rng-bit-generator.60 = (u64[2]{0}, u32[256]{0}) rng-bit-generator(u64[2]{0} %concatenate.59), algorithm=rng_default
  %get-tuple-element.62 = u64[2]{0} get-tuple-element((u64[2]{0}, u32[256]{0}) %rng-bit-generator.60), index=0
  %convert.77 = u64[] convert(s64[] %add.48)
  %reshape.79 = u64[1]{0} reshape(u64[] %convert.77)
  %constant.78 = u64[] constant(0)
  %reshape.80 = u64[1]{0} reshape(u64[] %constant.78)
  %concatenate.81 = u64[2]{0} concatenate(u64[1]{0} %reshape.79, u64[1]{0} %reshape.80), dimensions={0}
  %rng-bit-generator.82 = (u64[2]{0}, u32[256,256]{1,0}) rng-bit-generator(u64[2]{0} %concatenate.81), algorithm=rng_default
  %get-tuple-element.84 = u64[2]{0} get-tuple-element((u64[2]{0}, u32[256,256]{1,0}) %rng-bit-generator.82), index=0
  %constant.100 = s64[] constant(0)
  %p6.99 = f32[2,4,256,256]{3,2,1,0} parameter(6), sharding={devices=[4,1,1,1]0,1,2,3}
  %get-tuple-element.83 = u32[256,256]{1,0} get-tuple-element((u64[2]{0}, u32[256,256]{1,0}) %rng-bit-generator.82), index=1
  %constant.85 = u32[] constant(9)
  %broadcast.86 = u32[256,256]{1,0} broadcast(u32[] %constant.85), dimensions={}
  %shift-right-logical.87 = u32[256,256]{1,0} shift-right-logical(u32[256,256]{1,0} %get-tuple-element.83, u32[256,256]{1,0} %broadcast.86)
  %convert.88 = f32[256,256]{1,0} convert(u32[256,256]{1,0} %shift-right-logical.87)
  %constant.89 = f32[] constant(1.1920929e-07)
  %broadcast.90 = f32[256,256]{1,0} broadcast(f32[] %constant.89), dimensions={}
  %multiply.91 = f32[256,256]{1,0} multiply(f32[256,256]{1,0} %convert.88, f32[256,256]{1,0} %broadcast.90)
  %p4.53 = f32[] parameter(4), sharding={replicated}
  %p5.54 = f32[] parameter(5), sharding={replicated}
  %subtract.92 = f32[] subtract(f32[] %p4.53, f32[] %p5.54)
  %broadcast.93 = f32[256,256]{1,0} broadcast(f32[] %subtract.92), dimensions={}
  %multiply.94 = f32[256,256]{1,0} multiply(f32[256,256]{1,0} %multiply.91, f32[256,256]{1,0} %broadcast.93)
  %broadcast.95 = f32[256,256]{1,0} broadcast(f32[] %p5.54), dimensions={}
  %add.96 = f32[256,256]{1,0} add(f32[256,256]{1,0} %multiply.94, f32[256,256]{1,0} %broadcast.95)
  %reshape.97 = f32[1,256,256]{2,1,0} reshape(f32[256,256]{1,0} %add.96)
  %concatenate.98 = f32[1,256,256]{2,1,0} concatenate(f32[1,256,256]{2,1,0} %reshape.97), dimensions={0}
  %get-tuple-element.61 = u32[256]{0} get-tuple-element((u64[2]{0}, u32[256]{0}) %rng-bit-generator.60), index=1
  %constant.63 = u32[] constant(9)
  %broadcast.64 = u32[256]{0} broadcast(u32[] %constant.63), dimensions={}
  %shift-right-logical.65 = u32[256]{0} shift-right-logical(u32[256]{0} %get-tuple-element.61, u32[256]{0} %broadcast.64)
  %convert.66 = f32[256]{0} convert(u32[256]{0} %shift-right-logical.65)
  %constant.67 = f32[] constant(1.1920929e-07)
  %broadcast.68 = f32[256]{0} broadcast(f32[] %constant.67), dimensions={}
  %multiply.69 = f32[256]{0} multiply(f32[256]{0} %convert.66, f32[256]{0} %broadcast.68)
  %subtract.70 = f32[] subtract(f32[] %p4.53, f32[] %p5.54)
  %broadcast.71 = f32[256]{0} broadcast(f32[] %subtract.70), dimensions={}
  %multiply.72 = f32[256]{0} multiply(f32[256]{0} %multiply.69, f32[256]{0} %broadcast.71)
  %broadcast.73 = f32[256]{0} broadcast(f32[] %p5.54), dimensions={}
  %add.74 = f32[256]{0} add(f32[256]{0} %multiply.72, f32[256]{0} %broadcast.73)
  %reshape.75 = f32[1,256]{1,0} reshape(f32[256]{0} %add.74)
  %concatenate.76 = f32[1,256]{1,0} concatenate(f32[1,256]{1,0} %reshape.75), dimensions={0}
  %constant.39 = f32[] constant(0)
  %reshape.40 = f32[1,1,1,1,1]{4,3,2,1,0} reshape(f32[] %constant.39)
  %broadcast.41 = f32[1,1,1,1,1]{4,3,2,1,0} broadcast(f32[1,1,1,1,1]{4,3,2,1,0} %reshape.40), dimensions={0,1,2,3,4}
  %reshape.42 = f32[1]{0} reshape(f32[1,1,1,1,1]{4,3,2,1,0} %broadcast.41)
  %broadcast.43 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[1]{0} %reshape.42), dimensions={0}
  %constant.34 = f32[] constant(0)
  %reshape.35 = f32[1,1,1,1,1]{4,3,2,1,0} reshape(f32[] %constant.34)
  %broadcast.36 = f32[1,1,1,1,1]{4,3,2,1,0} broadcast(f32[1,1,1,1,1]{4,3,2,1,0} %reshape.35), dimensions={0,1,2,3,4}
  %reshape.37 = f32[1]{0} reshape(f32[1,1,1,1,1]{4,3,2,1,0} %broadcast.36)
  %broadcast.38 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[1]{0} %reshape.37), dimensions={0}
  %constant.29 = f32[] constant(0)
  %reshape.30 = f32[1,1,1,1,1]{4,3,2,1,0} reshape(f32[] %constant.29)
  %broadcast.31 = f32[1,1,1,1,1]{4,3,2,1,0} broadcast(f32[1,1,1,1,1]{4,3,2,1,0} %reshape.30), dimensions={0,1,2,3,4}
  %reshape.32 = f32[1]{0} reshape(f32[1,1,1,1,1]{4,3,2,1,0} %broadcast.31)
  %broadcast.33 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[1]{0} %reshape.32), dimensions={0}
  %constant.24 = f32[] constant(0)
  %reshape.25 = f32[1,1,1,1,1]{4,3,2,1,0} reshape(f32[] %constant.24)
  %broadcast.26 = f32[1,1,1,1,1]{4,3,2,1,0} broadcast(f32[1,1,1,1,1]{4,3,2,1,0} %reshape.25), dimensions={0,1,2,3,4}
  %reshape.27 = f32[1]{0} reshape(f32[1,1,1,1,1]{4,3,2,1,0} %broadcast.26)
  %broadcast.28 = f32[1,2,4,256,256]{4,3,2,1,0} broadcast(f32[1]{0} %reshape.27), dimensions={0}
  %constant.19 = f32[] constant(0)
  %reshape.20 = f32[1,1,1,1]{3,2,1,0} reshape(f32[] %constant.19)
  %broadcast.21 = f32[1,1,1,1]{3,2,1,0} broadcast(f32[1,1,1,1]{3,2,1,0} %reshape.20), dimensions={0,1,2,3}
  %reshape.22 = f32[1]{0} reshape(f32[1,1,1,1]{3,2,1,0} %broadcast.21)
  %broadcast.23 = f32[1,2,4,256]{3,2,1,0} broadcast(f32[1]{0} %reshape.22), dimensions={0}
  %constant.14 = f32[] constant(0)
  %reshape.15 = f32[1,1,1,1]{3,2,1,0} reshape(f32[] %constant.14)
  %broadcast.16 = f32[1,1,1,1]{3,2,1,0} broadcast(f32[1,1,1,1]{3,2,1,0} %reshape.15), dimensions={0,1,2,3}
  %reshape.17 = f32[1]{0} reshape(f32[1,1,1,1]{3,2,1,0} %broadcast.16)
  %broadcast.18 = f32[1,2,4,256]{3,2,1,0} broadcast(f32[1]{0} %reshape.17), dimensions={0}
  %constant.9 = f32[] constant(0)
  %reshape.10 = f32[1,1,1]{2,1,0} reshape(f32[] %constant.9)
  %broadcast.11 = f32[1,1,1]{2,1,0} broadcast(f32[1,1,1]{2,1,0} %reshape.10), dimensions={0,1,2}
  %reshape.12 = f32[1]{0} reshape(f32[1,1,1]{2,1,0} %broadcast.11)
  %broadcast.13 = f32[1,256,256]{2,1,0} broadcast(f32[1]{0} %reshape.12), dimensions={0}
  %constant.4 = f32[] constant(0)
  %reshape.5 = f32[1,1,1]{2,1,0} reshape(f32[] %constant.4)
  %broadcast.6 = f32[1,1,1]{2,1,0} broadcast(f32[1,1,1]{2,1,0} %reshape.5), dimensions={0,1,2}
  %reshape.7 = f32[1]{0} reshape(f32[1,1,1]{2,1,0} %broadcast.6)
  %broadcast.8 = f32[1,2048,256]{2,1,0} broadcast(f32[1]{0} %reshape.7), dimensions={0}
  %p2.3 = f32[2,4,256,256]{3,2,1,0} parameter(2), sharding={devices=[4,1,1,1]0,1,2,3}
  %p1.2 = f32[2,4,256,256]{3,2,1,0} parameter(1), sharding={devices=[4,1,1,1]0,1,2,3}
  %p0.1 = f32[2,4,256,256]{3,2,1,0} parameter(0), sharding={devices=[4,1,1,1]0,1,2,3}
  %call.280 = (s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) call(s64[] %constant.100, f32[2,4,256,256]{3,2,1,0} %p6.99, f32[1,256,256]{2,1,0} %concatenate.98, f32[1,256]{1,0} %concatenate.76, f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.43, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.38, f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.33, f32[1,2,4,256,256]{4,3,2,1,0} %broadcast.28, f32[1,2,4,256]{3,2,1,0} %broadcast.23, f32[1,2,4,256]{3,2,1,0} %broadcast.18, /*index=10*/f32[1,256,256]{2,1,0} %broadcast.13, f32[1,2048,256]{2,1,0} %broadcast.8, f32[2,4,256,256]{3,2,1,0} %p2.3, f32[2,4,256,256]{3,2,1,0} %p1.2, f32[2,4,256,256]{3,2,1,0} %p0.1), to_apply=%scan.262
  %get-tuple-element.281 = s64[] get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=0
  %get-tuple-element.283 = f32[1,256,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=2
  %get-tuple-element.284 = f32[1,256]{1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=3
  %get-tuple-element.285 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=4
  %get-tuple-element.286 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=5
  %get-tuple-element.287 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=6
  %get-tuple-element.288 = f32[1,2,4,256,256]{4,3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=7
  %get-tuple-element.289 = f32[1,2,4,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=8
  %get-tuple-element.290 = f32[1,2,4,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=9
  %get-tuple-element.291 = f32[1,256,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=10
  %get-tuple-element.292 = f32[1,2048,256]{2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=11
  %get-tuple-element.293 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=12
  %get-tuple-element.294 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=13
  %get-tuple-element.295 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=14
  %get-tuple-element.282 = f32[2,4,256,256]{3,2,1,0} get-tuple-element((s64[], f32[2,4,256,256]{3,2,1,0}, f32[1,256,256]{2,1,0}, f32[1,256]{1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, /*index=5*/f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256,256]{4,3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, f32[1,2,4,256]{3,2,1,0}, /*index=10*/f32[1,256,256]{2,1,0}, f32[1,2048,256]{2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}, f32[2,4,256,256]{3,2,1,0}) %call.280), index=1
  ROOT %tuple.296 = (f32[2,4,256,256]{3,2,1,0}) tuple(f32[2,4,256,256]{3,2,1,0} %get-tuple-element.282)
}




############### End scan ###############




Output with scan tensor([[[[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]],

         [[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]],

         [[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]],

         [[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]]],


        [[[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]],

         [[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]],

         [[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]],

         [[ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          ...,
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527],
          [ 0.0563, -0.0460,  0.0449,  ...,  0.0434, -0.0548, -0.0527]]]],
       device='xla:0', grad_fn=<ScanBackward>)
.
----------------------------------------------------------------------
Ran 2 tests in 2.658s

```
