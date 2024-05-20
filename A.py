# return body(iter, *input_ouput_values)

(torch310) root@23587be5ee07:/pytorch/xla# PJRT_DEVICE=TPU python test/test_fori_loop_with_while_loop_simple_add_dispatch_in_torch.py 
WhileLoopTest.test_while_loop_tpu_subtraction_s32

body computation: !!!!!!!!!
HloModule PyLoweringContext.14.21, entry_computation_layout={(
    (s32[], s32[], s32[], s32[]))
    ->
    (s32[], s32[], s32[])}

%PyLoweringContext.6 
(p0.9: s32[], UnusedArgumentsPlaceholder.16: s32[], UnusedArgumentsPlaceholder.17: s32[], UnusedArgumentsPlaceholder.18: s32[]) 
-> 
(s32[], s32[], s32[]) 
{
  %UnusedArgumentsPlaceholder.16 = s32[] parameter(1)
  %UnusedArgumentsPlaceholder.17 = s32[] parameter(2)
  %UnusedArgumentsPlaceholder.18 = s32[] parameter(3)
  //// iter
  %p0.9 = s32[] parameter(0)
  //// 1
  %constant.8 = s32[] constant(1)
  %constant.7 = s32[] constant(1)
  %multiply.10 = s32[] multiply(s32[] %constant.8, s32[] %constant.7)

  //// iter - 1
  %subtract.11 = s32[] subtract(s32[] %p0.9, s32[] %multiply.10)

  //// 1
  %constant.13 = s32[] constant(1)
  %constant.12 = s32[] constant(1)
  %multiply.14 = s32[] multiply(s32[] %constant.13, s32[] %constant.12)

  //// iter - 1
  %subtract.15 = s32[] subtract(s32[] %p0.9, s32[] %multiply.14)

  ////                                        (iter-1,             iter,        iter-1)
  ROOT %tuple.19 = (s32[], s32[], s32[]) tuple(s32[] %subtract.11, s32[] %p0.9, s32[] %subtract.15)
}

ENTRY %PyLoweringContext.14.21 (in.1: (s32[], s32[], s32[], s32[])) -> (s32[], s32[], s32[]) {
  %in.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=0
  %get-tuple-element.3 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=2
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=3
  ROOT %call.20 = (s32[], s32[], s32[]) call(s32[] %get-tuple-element.2, s32[] %get-tuple-element.3, s32[] %get-tuple-element.4, s32[] %get-tuple-element.5), to_apply=%PyLoweringContext.6
}


cond computation: !!!!!!!!!
//// iter > 0

HloModule PyLoweringContext.8.15, entry_computation_layout={(
    (s32[], s32[], s32[], s32[]))
    ->
    pred[]}

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
s}

ENTRY %PyLoweringContext.8.15 (in.1: (s32[], s32[], s32[], s32[])) -> pred[] {
  %in.1 = (s32[], s32[], s32[], s32[]) parameter(0)
  %get-tuple-element.2 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=0
  %get-tuple-element.3 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=1
  %get-tuple-element.4 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=2
  %get-tuple-element.5 = s32[] get-tuple-element((s32[], s32[], s32[], s32[]) %in.1), index=3
  ROOT %call.14 = pred[] call(s32[] %get-tuple-element.2, s32[] %get-tuple-element.3, s32[] %get-tuple-element.4, s32[] %get-tuple-element.5), to_apply=%PyLoweringContext.6
}


F0516 21:59:47.227707 1514341 debug_macros.h:20] Non-OK-status: status.status() status: INVALID_ARGUMENT: 
The parameter of condition and body, the result of the body, and init must all have the same shape; got 

Condition: (
    in: 
    (s32[], s32[], s32[], s32[])) 
    -> 
    pred[]; 
    
body: (
    in: 
    (s32[], s32[], s32[], s32[])) 
    -> 
    (s32[], s32[], s32[]); 

init: 
    (s32[], s32[], s32[], s32[])


