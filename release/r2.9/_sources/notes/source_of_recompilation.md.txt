# Source of recompilations in torch_xla

## Let’s first start with some facts/constraints:

1. Graph compilations in XLA are pretty expensive.
2. XLA handles static shape only. In other words, even for the same IR graph, XLA recompiles when input shape changes.
3. Recompilations hurts torch_xla perf a lot when it happens, and it’s hard to understand and debug from a normal python user POV.

Often when recompilation happens we say we just need dynamic shape support and then rest assured that when dynamic shape is supported in the future, all the recompilations will be magically gone. But this is not true, XLA now has pretty good bounded dynamic shapes coverage already, but we still see recompilations and they are expected.

***This doc aims to provide a detailed explanation of a few common sources of recompilations, and what do we need to get rid of them.  It will mainly focus on explaining the problem to beginners without any context. To make it easy to understand, the “solutions” proposed here may rely on impractical assumptions.* **

## #1. From input dataset.

Yes it’s pretty common that input dataset contains examples with different shapes, e.g. sentences with varying length or images with different sizes. Without normalization, it’ll cause recompilation for every new input shape.

Tensorflow graph mode users are more used to do padding/bucketization (`tf.pad`) to normalize input shapes to one or a few buckets. But this is kinda anti-pattern for PyTorch eager frontend users (which is the same user lazy tensor frontend is trying to target) since different input shapes just doesn’t matter for eager CPU/CUDA backend.

**Proposed workaround:** okay now let’s say we can work around this problem by teaching our users to do padding/bucketization (it’s hard in practice :P). What’s next?

## #2. From operator output

There are certain operators semantically are data-dependent and produce dynamic shape outputs: e.g. `torch.nonzero` returns indices of nonzero elements in its input tensor. So even your input tensors to this operator always have the same shape, it might produce different shape outputs and cause recompilations.

### 2.1 Bounded dynamic shape can fix the case when you use the tensor with dynamic shape as a Tensor, without querying its real dimension.

**Proposed workaround:** let’s say now XLA supports bounded dynamic shape for all operators, is it good enough?

* by bounded dynamic shape it means we can pad the tensor to a theoretical max, trading more memory usage for less recompilation/faster speed.

Well, sort of. Let’s see the following example:


```
a = torch.tensor([1, 2, 0, 1, 3], device='xla')
b = torch.nonzero(a)
c = b * 2
d = c + 1
print(torch_xla._XLAC._get_xla_tensors_text([d]))
```

In the example above every node below `b` in the graph (namely `c, d` and everything depend on them) will have dynamic shape, it’s pretty obvious that `b` has dynamic shape in dimension 0 as shown below:


```
  %9 = (s64[<=5,1]{1,0}, s64[]) aten::nonzero(%8), num_outputs=2 # b
  %10 = s64[5,1]{1,0} aten::mul(%9.0, %3) # c
  %11 = s64[5,1]{1,0} aten::add(%10, %2), ROOT=0 # d
```

Although it's not shown directly in the graph, `c & d` indeed also have dynamic shape (in other words, [5, 1] is just padded shape and it's masked).

```
print(torch_xla._XLAC._get_xla_tensor_dimension_size(d, 0)) # prints 4 instead of 5
```

You can see that in this case as long as the input tensor `a` has shape `[5]` we only compile the graph once. Bounded dynamic shape support helped!

### 2.2 what if real dimension is queried on a tensor with dynamic shape?

This is actually pretty commonly used since not all PyTorch computation are done in the form of Tensors.

For example, `tensor.size()` in PyTorch returns a tuple of ints instead of a Tensor of dtype=int. When `tensor` is a dynamic shape tensor, this op basically forces XLA to cut the graph and evaluate so that we can return the correct scalar (otherwise it’ll just return the padded shape which is wrong).

What’s made it worse is that many PyTorch takes scalar inputs as well. After you do `s = tensor.size(0)` and use `s` in other operators it also becomes a dynamic source. In this case we probably know how to pad it and its upper bound, but we cannot do it since it’s not even a Tensor!


```
 a = torch.tensor([1, 2, 0, 1, 3], device='xla')
 b = torch.nonzero(a)
 s = a.size(0) # evaluation happens! nit: we use size() for simplicity, the actual API is _get_xla_tensor_dimension_size.
 c = torch.rand(s, device='xla') # c can be of any shape between [0, 5] which causes more recompilations!
 d = c + 1
```

So this one is actually hard to solve without PyTorch frontend’s help. What do we need?

In short, we need a Tensor world!

For example,

* `tensor.size()` should return a Tensor so that it can be a Tensor with dynamic shape and kept in the graph without early evaluation.
* Tensor accessor, e.g. for 2D tensor, `tensor[0][0]` now returns a value but this need to return a tensor as well.
* Implicitly this means all operators currently taking int/float/double as input need a Tensor overload as well. THIS IS A BIG ASK as it can easily explode our operator set.
    * It’s easier if we can make scalar to Tensor conversion really cheap so that we can only care about the Tensor overload.
    * In practice not all ops takes scalars from previous computation, so we’ve been adding Tensor variants by ad-hoc requests.
    * This is also a common ask from tracing base approaches I think.

Okay now that we assume every op in PyTorch has a Tensor verison we need, are we done?

## #3. From control flow

No! We actually only solved the problem without data dependent control flow...

See the example below:

```
if x[0][0] == 3:
  bla
else:
  blabla
```

Even if `x[0][0]` was a Tensor, we need to execute/materialize its value for python interpreter to proceed. And different branch choices in multiple control flows combined means we have a lot of graph to compile as well!

For now we just have no way to fix this. To fix it we need to lower the control flow from python to graph! Without too much thinking in implementation we can do this in two ways:

* ask users to explicitly use a control flow op instead of python if/else/while/for. This is currently supported as [customized API in torch_xla](https://github.com/pytorch/xla/blob/master/torch_xla/core/xla_builder.py#L563-L574) but not widely adopted in users’ code. (python users are used to if/else/for and it’s hard to switch them to a uglier API unless there’s a huge perf win).
* parse python source. code to get the control flow statement automatically. This is like Torchscript and somehow merge the torchscripted graph into the lazily trace graph properly (including shape info etc). I haven’t thought through the steps of how to implement this indeed :P

But either solution above requires non-trivial amount of effort, either on user side or on the framework side. That’s why we currently just take the hit of early evaluation & multiple compilations as a short term solution given the bandwidth we have.

Okay so now we assume that also have control flow lowered in the graph automagically, are we gold?

YES! Now you have your whole computation represented in a graph of Tensor operations, including control flow so that compilers can now consume and do their smart tricks! But tbh at this point your program is no longer very PyTorch-y.


## Conclusion:

There’re actually multiple sources of recompilation and bounded dynamic shape support cannot solve all of them. The proposed workarounds in this doc are definitely sometimes impractical, and there might be better ways to fix each source properly that I’m totally unaware of. But I hope as we keep smashing our way to an ideal lazy tensor stack in this doc, it’s now easier for you understand what’re the remaining blockers ahead of us.


## Appendix:

1. NNC uses symbolic shapes, does that help?

Yes but partially. By having symbolic shape, your compilation optimization no longer requires concrete shape values. In other words your generated kernel are more general than XLA’s static shape ones.

And which exactly problem does it help?

It helps with cases like #1 and #2.1.


```
shape [3, 5] -> add -> transpose -> ... -> mul
shape [6, 2] -> add -> transpose -> ... -> mul

# with symbolic shape
shape [x, y] -> add -> transpose -> ... -> mul
```

With symbolic shape your generated kernel doesn’t recompile as XLA does with static shapes.

XLA solves this problem in the other way, by using padding/bucketization (for #1) and bounded dynamic shape (for #2.1).

Brian Hirsh(@bdhirsh) asked some really good questions in the comment, moving here to make them more visible:

2. Is it worth sticking a TORCH_WARN in the XLA kernels of ops that produce data-dependent output shapes?

Yea torch_warn is useful in telling users "hey your program won't run blazing fast". But for these data dependent ops, there isn't an easy rewrite for them unless users change the logic in their model. (another example is torch.unique())

3. How ops like nonzero impact our ability to devirtualize sizes()? If we want to devirtualize sizes(), we’ll need to be able to eagerly compute sizes for each op - won’t that mean we’re forced to evaluate the graph every time we hit an op like nonzero? Vs. right now, it sounds like we don't actually force an evaluation when a user calls nonzero()?

Yea great question! So in the current form it’s not a hard blocker since size() on XLA Tensors doesn’t carry source of truth size information. As shown in the example, the source of truth lives in IRValue and can be retrieved by `_get_xla_tensor_dimension_size` only. So if we decide to devirtualize size it’ll just enforce this discrepancy.

As a followup if we have `size()` return Tensor instead of values as mentioned in the proposed workarounds above. In that case size() won’t be able to devirtualize since it becomes an operator (taking in Tensor and produce Tensor, have different implementations for different backends.)

4. If I, e.g. call `torch.add(input, 1)` in a loop, where input varies in size from 1-1000, normally we would have to compile 1000 different graphs - but with dynamic shapes, it sounds like XLA will internally be able to generate a single graph where it says “use this graph if the input size is <=1000”. My question is: is “dynamic shape” a property of just the graph? Or of both the graph and the input. I.e. if my code were instead calling `x = torch.add(input, 1); x.sizes()` in a loop, does x have a dynamic shape at this point, meaning we’d need to run the graph to get the sizes? Or are we able to make it an eagerly computed property even in the presence of graphs with dynamic shapes.

Yea in this case you'll compile 1000 different graphs. Dynamic shapes means its input has dynamic dimension in it. So when you query `x.sizes()` (currently need use get_dimention_size to get the correct size) it'll trigger *execution* (since the size didn't change it doesn't trigger recompilation). Without the line accessing size, it won't trigger any recompilation/execution when input has dynamic dimension.

5. Would an alternative of making control flow available in the graph be just to come up with a way to ensure that XLA graphs don't include control flow? i.e. if we have a model with a single conditional in the middle, then get XLA to produce 3 graphs: 1 for everything before the conditional, 1 for the if branch, and 1 for the else branch. That would mean you don't get the exponential blowup of new graphs for every combination of paths taken, but (a) the graphs are smaller and provide fewer optimization opportunities, and (b) it would probably be pretty non-trivial to get XLA to recognize where a conditional path is taken.

Great point! So if we could break them up into smaller graphs it's indeed feasible. But in practice this pattern is annoying:

```
y = <some computation>
x = y + 2
if x[0] == 2 :
  z = y +1
else:
  z = y - 1
```

Note you'll evaluate x using a subgraph when you hit control flow, but there might be previous variable included in the branch computation as well (like` y` is just one node smaller than x, but it wasn't materizalized when you evaluate `x`). So you're actually evaluating 1 small graph and two big graphs for this example. And with more control flow involved, y could get updated in multiple branches which still produces different combo of large graphs.

