# return modified res

(torch310) root@23587be5ee07:/pytorch/xla# PJRT_DEVICE=TPU python test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py 
WhileLoopTest.test_while_loop_tpu_subtraction_s32
body computation: !!!!!!!!!
HloModule PyLoweringContext.15.22, entry_computation_layout={(
    (s32[], s32[], s32[], s32[]))
    ->
    (s32[], s32[], s32[], s32[])}

%PyLoweringContext.6 
(p0.9: s32[], UnusedArgumentsPlaceholder.17: s32[], UnusedArgumentsPlaceholder.18: s32[], UnusedArgumentsPlaceholder.19: s32[]) 
-> 
(s32[], s32[], s32[], s32[]) 
{
  %UnusedArgumentsPlaceholder.17 = s32[] parameter(1)
  %UnusedArgumentsPlaceholder.18 = s32[] parameter(2)
  %UnusedArgumentsPlaceholder.19 = s32[] parameter(3)

  //// iter
  %p0.9 = s32[] parameter(0)

  //// 1
  %constant.8 = s32[] constant(1)
  %constant.7 = s32[] constant(1)
  %multiply.10 = s32[] multiply(s32[] %constant.8, s32[] %constant.7)

  //// iter - 1
  %subtract.11 = s32[] subtract(s32[] %p0.9, s32[] %multiply.10)

  //// 1
  %constant.12 = s32[] constant(1)
  %constant.14 = s32[] constant(1)
  %constant.13 = s32[] constant(1)
  %multiply.15 = s32[] multiply(s32[] %constant.14, s32[] %constant.13)

  //// iter - 1
  %subtract.16 = s32[] subtract(s32[] %p0.9, s32[] %multiply.15)

  ////                                               (iter-1,             1,                  iter,        iter-1)
  ROOT %tuple.20 = (s32[], s32[], s32[], s32[]) tuple(s32[] %subtract.11, s32[] %constant.12, s32[] %p0.9, s32[] %subtract.16)
}

ENTRY %PyLoweringContext.15.22 (in.1: (s32[], s32[], s32[], s32[])) -> (s32[], s32[], s32[], s32[]) {
  %in.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=0
  %get-tuple-element.3 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=2
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=3
  ROOT %call.21 = (s32[], s32[], s32[], s32[]) call(s32[] %get-tuple-element.2, s32[] %get-tuple-element.3, s32[] %get-tuple-element.4, s32[] %get-tuple-element.5), to_apply=%PyLoweringContext.6
}


cond computation: !!!!!!!!!
//// iter > 0

HloModule PyLoweringContext.8.15, entry_computation_layout={((s32[], s32[], s32[], s32[]))->pred[]}

%PyLoweringContext.6 
(p0.8: s32[], UnusedArgumentsPlaceholder.11: s32[], UnusedArgumentsPlaceholder.12: s32[], UnusedArgumentsPlaceholder.13: s32[]) 
-> 
pred[] 
{
  %p0.8 = s32[] parameter(0)
  %convert.9 = s64[] convert(s32[] %p0.8)
  %constant.7 = s64[] constant(0)
  ROOT %compare.10 = pred[] compare(s64[] %convert.9, s64[] %constant.7), direction=GT
  %UnusedArgumentsPlaceholder.11 = s32[] parameter(1)
  %UnusedArgumentsPlaceholder.12 = s32[] parameter(2)
  %UnusedArgumentsPlaceholder.13 = s32[] parameter(3)
}

ENTRY %PyLoweringContext.8.15 (in.1: (s32[], s32[], s32[], s32[])) -> pred[] {
  %in.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=0
  %get-tuple-element.3 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=2
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=3
  ROOT %call.14 = pred[] call(s32[] %get-tuple-element.2, s32[] %get-tuple-element.3, s32[] %get-tuple-element.4, s32[] %get-tuple-element.5), to_apply=%PyLoweringContext.6
}


res:  [tensor(0, device='xla:0', dtype=torch.int32), tensor(1, device='xla:0', dtype=torch.int32), tensor(1, device='xla:0', dtype=torch.int32), tensor(0, device='xla:0', dtype=torch.int32)]
expected:  (tensor(0, device='xla:0', dtype=torch.int32), tensor(10, device='xla:0', dtype=torch.int32), tensor(9, device='xla:0', dtype=torch.int32))
.
----------------------------------------------------------------------
Ran 1 test in 3.612s

OK