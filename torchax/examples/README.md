## Intro

This readme will have a subsection for every example *.py file.

Please follow the instructions in [README.md](../README.md) to install torchax,
then install requirements for all of the examples with

```bash
pip install -r requirements.txt
```



## basic_training.py

This file constructed by first copy & paste code fragments from this pytorch training tutorial:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

Then adding few lines of code that serves the purpose of moving `torch.Tensor` into
`XLA devices`.

Example:

```python
state_dict = pytree.tree_map_only(torch.Tensor,
    torchax.tensor.move_to_device, state_dict)
```

This fragment moves the state_dict to XLA devices; then the state_dict is passed
back to model via `load_state_dict`.

Then, you can train the model. This shows what is minimum to train a model on XLA
devices. The perf is not as good because we didn't use `jax.jit`, this is intentional
as it is meant to showcase the minimum code change.

Example run:
```bash
(xla2) hanq-macbookpro:examples hanq$ python basic_training.py
Training set has 60000 instances
Validation set has 10000 instances
Bag  Dress  Sneaker  T-shirt/top
tensor([[0.8820, 0.3807, 0.3010, 0.9266, 0.7253, 0.9265, 0.0688, 0.4567, 0.7035,
         0.2279],
        [0.3253, 0.1558, 0.1274, 0.2776, 0.2590, 0.4169, 0.1881, 0.7423, 0.4561,
         0.5985],
        [0.5067, 0.4514, 0.9758, 0.6088, 0.7438, 0.6811, 0.9609, 0.3572, 0.4504,
         0.8738],
        [0.1850, 0.1217, 0.8551, 0.2120, 0.9902, 0.7623, 0.1658, 0.6980, 0.3086,
         0.5709]])
tensor([1, 5, 3, 7])
Total loss for this batch: 2.325265645980835
EPOCH 1:
  batch 1000 loss: 1.041275198560208
  batch 2000 loss: 0.6450189483696595
  batch 3000 loss: 0.5793989677671343
  batch 4000 loss: 0.5170258888280951
  batch 5000 loss: 0.4920090722264722
  batch 6000 loss: 0.48910293977567926
  batch 7000 loss: 0.48058812761632724
  batch 8000 loss: 0.47159107415075413
  batch 9000 loss: 0.4712311488997657
  batch 10000 loss: 0.4675815168160479
  batch 11000 loss: 0.43210567891132085
  batch 12000 loss: 0.445208148030797
  batch 13000 loss: 0.4119230824254337
  batch 14000 loss: 0.4190662656680215
  batch 15000 loss: 0.4094535468676477
LOSS train 0.4094535468676477 valid XLA
```

## basic_training_jax.py

This file constructed by first copy & paste code fragments from this pytorch training tutorial:
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

Then replacing torch optimizer with `optax` optimizer; and use `jax.grad` for
gradient instead of `torch.Tensor.backward()`.

Then, you can train the model using jax ecosystem's training loop. This is meant to
showcase how easy is to integrate with Jax.

Example run:
```bash
(xla2) hanq-macbookpro:examples hanq$ python basic_training_jax.py
Training set has 60000 instances
Validation set has 10000 instances
Pullover  Ankle Boot  Pullover  Ankle Boot
tensor([[0.5279, 0.8340, 0.3131, 0.8608, 0.3668, 0.6192, 0.7453, 0.3261, 0.8872,
         0.1854],
        [0.7414, 0.8309, 0.8127, 0.8866, 0.2475, 0.2664, 0.0327, 0.6918, 0.6010,
         0.2766],
        [0.3304, 0.9135, 0.2762, 0.6737, 0.0480, 0.6150, 0.5610, 0.5804, 0.9607,
         0.6450],
        [0.9464, 0.9439, 0.3122, 0.1814, 0.1194, 0.5012, 0.2058, 0.1170, 0.7377,
         0.7453]])
tensor([1, 5, 3, 7])
Total loss for this batch: 2.4054245948791504
EPOCH 1:
  batch 1000 loss: 1.0705260595591972
  batch 2000 loss: 1.0997755021179327
  batch 3000 loss: 1.0186579653513108
  batch 4000 loss: 0.9090727646966116
  batch 5000 loss: 0.8309370622411024
  batch 6000 loss: 0.8702225417760783
  batch 7000 loss: 0.8750176187023462
  batch 8000 loss: 0.9652624803795453
  batch 9000 loss: 0.8688667197711766
  batch 10000 loss: 0.8021814124770199
  batch 11000 loss: 0.8000540231048071
  batch 12000 loss: 0.9150884484921057
  batch 13000 loss: 0.819690621060171
  batch 14000 loss: 0.8569030471532278
  batch 15000 loss: 0.8740896808278603
LOSS train 0.8740896808278603 valid 2.3132264614105225
```
