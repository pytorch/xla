from __future__ import print_function

import collections
import queue
import threading
import time
import torch
import torch.nn as nn
import torch_xla
import torch_xla_py.utils as xu
import torch_xla_py.keyd_queue as kq

MultiBatch = collections.namedtuple('MultiBatch',
                                    ['batch_number', 'inputs', 'targets'])


class LinearIndex(object):
    def __init__(self, index):
        self.index = index


class ToXlaTensorArena(object):
    def __init__(self):
        self._tensors = []
        self._devices = []
        self._converted_tensors = None

    def add(self, tensor, device=None):
        if self._tensors:
            assert type(self._tensors[0]) == type(tensor)
        self._tensors.append(tensor)
        if device is not None:
            self._devices.append(device)
        return LinearIndex(len(self._tensors) - 1)

    def convert(self):
        if not self._tensors:
            return
        if type(self._tensors[0]) == torch.Tensor:
            assert self._devices
            self._converted_tensors = torch_xla._XLAC._xla_create_tensors(
                self._tensors, self._devices)
        elif type(self._tensors[0]) == torch_xla._XLAC.XLATensor:
            assert not self._devices
            self._converted_tensors = torch_xla._XLAC._xla_to_tensors(
                self._tensors)
        else:
            self._converted_tensors = self._tensors

    def get_converted_tensor(self, lindex):
        assert isinstance(lindex, LinearIndex)
        assert self._converted_tensors is not None
        assert lindex.index < len(self._converted_tensors)
        return self._converted_tensors[lindex.index]


def _collect_tensors(arena, collect_type, inputs, devices=None, device=""):
    if type(inputs) == collect_type:
        return arena.add(inputs, device=device)
    if isinstance(inputs, (list, tuple)):
        tensors = []
        for i, input in enumerate(inputs):
            if devices is not None:
                # The first dimension, if devices is specified, is the replica
                # dimension, and we assign every nested tensor to the proper
                # replica device.
                assert len(devices) > i
                device = devices[i]
            tensors.append(_collect_tensors(arena, collect_type, input,
                                            device=device))
        return tuple(tensors)
    return inputs


def _replace_tensors(arena, tensors):
    if isinstance(tensors, LinearIndex):
        return arena.get_converted_tensor(tensors)
    if isinstance(tensors, (list, tuple)):
        new_tensors = []
        for tensor in tensors:
            new_tensors.append(_replace_tensors(arena, tensor))
        return tuple(new_tensors)
    return tensors


def forward_passes(graph):
    torch._C._jit_pass_canonicalize_ops(graph)
    torch_xla._XLAC._jit_pass_set_mat_mul_output_shape(graph)
    torch_xla._XLAC._jit_pass_insert_explicit_expand(graph)
    torch_xla._XLAC._jit_pass_eval_static_size(graph)
    torch._C._jit_pass_constant_propagation(graph)
    torch_xla._XLAC._jit_pass_replace_untraced_operators(graph)
    torch._C._jit_pass_dce(graph)


def backward_passes(graph):
    torch._C._jit_pass_specialize_undef(graph)
    torch_xla._XLAC._jit_pass_eval_static_size(graph)
    torch._C._jit_pass_constant_propagation(graph)
    torch_xla._XLAC._jit_pass_threshold_backward_peephole(graph)
    torch._C._jit_pass_dce(graph)


def convert_to_xla_tensors(inputs, devices=None):
    arena = ToXlaTensorArena()
    tensors = _collect_tensors(arena, torch.Tensor, inputs, devices=devices)
    arena.convert()
    return _replace_tensors(arena, tensors)


def convert_to_tensors(inputs):
    arena = ToXlaTensorArena()
    tensors = _collect_tensors(
        arena, torch_xla._XLAC.XLATensor, inputs, device=None)
    arena.convert()
    return _replace_tensors(arena, tensors)


def create_xla_model(model, inputs, num_cores=1, devices=None,
                     input_gradients=None, full_conv_precision=False):
    assert isinstance(inputs, (tuple, list))
    assert num_cores == 1 or num_cores == len(devices)
    replica_inputs = []
    for n in range(0, num_cores):
        replica_inputs.append(inputs)
    traced_model = torch.jit.trace(model, inputs)
    xla_model = torch_xla._XLAC.XlaModule(
        traced_model, use_full_conv_precision=full_conv_precision)
    inputs_xla = convert_to_xla_tensors(replica_inputs, devices=devices)
    if input_gradients is not None:
        xla_model.set_inputs_gardients(input_gradients)
    xla_model(*inputs_xla)
    return xla_model, traced_model


def update_optimizer_state(optimizer, name, value):
    for param_group in optimizer.param_groups:
        if not name in param_group:
            continue
        if callable(value):
            param_group[name] = value(param_group[name])
        else:
            param_group[name] = value


def read_multi_batch(train_loader_enumerator, batch_size, splits=1,
                     fused_mode=False):
    inputs = []
    targets = []
    batch_number = None
    splitno = 0
    for batch_idx, (data, target) in train_loader_enumerator:
        if data.size()[0] != batch_size:
            # The last batch size is inconsistent; XLA cannot handle it.
            break
        if batch_number is None:
            batch_number = int(batch_idx / splits)
        if fused_mode:
            inputs.append([data, target])
        else:
            inputs.append([data])
            targets.append(target)
        splitno += 1
        if splitno == splits:
            return MultiBatch(batch_number=batch_number, inputs=inputs,
                              targets=targets)


def zeros_like(p):
    if isinstance(p, torch_xla._XLAC.XLATensor):
        return torch.zeros(p.size(), dtype=p.dtype)
    return torch.zeros_like(p.data)


def extract_gradients(inputs, fill_fn=None):
    if isinstance(inputs, (torch.Tensor, torch_xla._XLAC.XLATensor)):
        grad = inputs.grad
        if grad is not None or fill_fn is None:
            return grad
        return fill_fn(inputs)
    if isinstance(inputs, (list, tuple)):
        grad_inputs = []
        for input in inputs:
            grad_inputs.append(extract_gradients(input, fill_fn=fill_fn))
        return tuple(grad_inputs)
    raise RuntimeError('Unable to extract gradients: {}'.format(type(inputs)))


# Run an XLA model with the given tensors.
def run_xla_model(xla_model, inputs, devices=None):
    return xla_model(*convert_to_xla_tensors(inputs, devices=devices))


def get_flat_tensors(xla_tensors):
    flat_xla_tensors = []
    for replica_xla_tensors in xla_tensors:
        for out in replica_xla_tensors:
            if isinstance(out, torch_xla._XLAC.XLATensor):
                flat_xla_tensors.append(out)
    return torch_xla._XLAC._xla_to_tensors(flat_xla_tensors)


# Compute the given loss function for the given XLA output tensors and the
# labels.
# Returns a tuple with the losses and the outputs converted to a Torch tensor.
def xla_loss(loss_fn, output_xla_tensors, labels):
    assert len(output_xla_tensors) == len(labels)
    flat_tensors = get_flat_tensors(output_xla_tensors)
    losses = []
    outputs = []
    flat_index = 0
    for i, replica_xla_tensors in enumerate(output_xla_tensors):
        replica_outputs = []
        for out in replica_xla_tensors:
            flat_tensors[flat_index].requires_grad = True
            replica_outputs.append(flat_tensors[flat_index])
            flat_index += 1
        replica_outputs.append(labels[i])
        losses.append(loss_fn(*replica_outputs))
        outputs.append(tuple(replica_outputs))
    return losses, tuple(outputs)


# Runs the backward pass for the given XLA model and the gradient outputs.
def xla_run_grad(xla_model, grad_outputs, devices=None):
    # Trace and symbolically differentiate
    grads_output_xla = convert_to_xla_tensors(grad_outputs, devices=devices)
    xla_model.backward(*grads_output_xla)


def category_eval_fn(loss_fn):
    def eval_fn(output, target):
        loss = loss_fn(output, target, reduction='sum').item()
        # Get the index of the max log-probability.
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        return loss, correct

    return eval_fn


class LoaderWrapper(object):
    def __init__(self, loader, prefetch_size, batch_size, num_cores=1,
                 devices=None, fused_mode=False):
        self._loader = loader
        self._prefetch_size = prefetch_size
        self._batch_size = batch_size
        self._num_cores = num_cores
        self._devices = list(devices) if devices else None
        self._fused_mode = fused_mode
        self._batch_number = 0
        self._done = False
        self._lock = threading.Lock()
        self._loader_queue = kq.Queue(maxsize=self._prefetch_size)
        self._queue = kq.KeydQueue(maxsize=self._prefetch_size)
        self._worker_count = 0
        thread = threading.Thread(target=self._loader_worker)
        thread.daemon = True
        thread.start()
        for _ in range(0, prefetch_size):
            thread = threading.Thread(target=self._worker)
            thread.daemon = True
            thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        item = self._queue.get(self._batch_number)
        if item is None:
            raise StopIteration
        self._batch_number += 1
        return self._batch_number - 1, item

    def close(self):
        self._done = True
        self._queue.close()
        self._loader_queue.close()

    def _up_workers(self, count):
        with self._lock:
            self._worker_count += count
            return self._worker_count

    def _loader_worker(self):
        inputs = []
        targets = []
        batch_number = 0
        for batch_idx, (data, target) in enumerate(self._loader):
            if data.size()[0] != self._batch_size or self._done:
                break
            if not inputs:
                batch_number = int(batch_idx / self._num_cores)
            if self._fused_mode:
                inputs.append([data, target])
            else:
                inputs.append([data])
                targets.append(target)
            if len(inputs) == self._num_cores:
                self._loader_queue.put((batch_number, (inputs, targets)))
                inputs = []
                targets = []
        self._loader_queue.close_write()

    def _worker(self):
        self._up_workers(1)
        while True:
            item = self._loader_queue.get()
            if item is None:
                break
            batch_number, (inputs, targets) = item
            inputs_xla = convert_to_xla_tensors(
                inputs, devices=self._devices)
            if targets:
                targets_xla = convert_to_xla_tensors(
                    targets, devices=self._devices)
            else:
                targets_xla = []
            self._queue.put(batch_number, (inputs_xla, targets_xla))
        if self._up_workers(-1) == 0:
            self._queue.close_write()


def _wrap_module(module, loss_fn):
    original_forward = module.forward

    def forward(input, target):
        # Return loss as ordinal 0, and orignal model output as ordinal 1.
        # The loss return allows the full fusion of the training graph, and the
        # original output is used for testing accuracy.
        output = original_forward(input)
        return loss_fn(output, target), output

    module.forward = forward
    return module


def _create_wrapped_model_backward_grads(model_fn, inputs, target):
    inputs_and_target = xu.list_copy_append(inputs, target)
    outputs = model_fn(*inputs_and_target)
    # Loss and Output.
    assert len(outputs) == 2
    loss = outputs[0]
    output = outputs[1]
    # The wrapped model function has a (loss, wrapped_model_ouput) output.
    # The gradient of the los WRT itself is one, and we are not interested
    # in the wrapped_model_ouput componenent.
    ones = torch.ones_like(loss)
    zeros = torch.zeros_like(output)
    return [ones, zeros]


class XlaModel(object):
    def __init__(self, model, inputs, target=None, loss_fn=None, num_cores=1,
                 devices=None, loader_prefetch=4, full_conv_precision=False):
        self._model = model
        self._model_fn = _wrap_module(model, loss_fn) if loss_fn else model
        self._loss_fn = loss_fn
        self._num_cores = num_cores
        self._devices = list(devices) if devices else None
        self._loader_prefetch = loader_prefetch
        self._epoch = 0
        if loss_fn:
            assert target is not None
            loss_output_grads = _create_wrapped_model_backward_grads(
                self._model_fn, inputs, target)
            inputs_and_target = xu.list_copy_append(inputs, target)
            self._xla_model, self._traced_model = create_xla_model(
                self._model_fn, inputs_and_target, num_cores=self._num_cores,
                devices=devices, input_gradients=loss_output_grads,
                full_conv_precision=full_conv_precision)
        else:
            self._xla_model, self._traced_model = create_xla_model(
                self._model_fn, inputs, num_cores=self._num_cores,
                devices=devices, full_conv_precision=full_conv_precision)

    def traced_model(self):
        return self._traced_model

    def _get_backward_grads(self, outputs):
        if self._loss_fn is None:
            # If this is the legacy API, the user has run loss.backward() and
            # the output passed here will have their gradient set.
            return extract_gradients([outputs], fill_fn=zeros_like)
        return []

    def __call__(self, *args):
        if self._loss_fn is None and self._num_cores == 1:
            # Internally the interface to the XLA module is always in replicated
            # mode, where num_replicas=1 is just a case of replication.
            outputs = run_xla_model(self._xla_model, [args],
                                    devices=self._devices)
            # If in legacy-API mode, convert the XLA tensor directly to PyTorch
            # tensor.
            return convert_to_tensors(outputs[0])
        assert len(args) == self._num_cores
        return run_xla_model(self._xla_model, args, devices=self._devices)

    def backward(self, outputs):
        xla_run_grad(self._xla_model, self._get_backward_grads(outputs),
                     devices=self._devices)

    def parameters(self):
        return self._xla_model.parameters()

    def parameters_list(self):
        return xu.flatten_nested_tuple(self.parameters())

    def parameters_buffers(self):
        return self._xla_model.parameters_buffers()

    def parameters_buffers_list(self):
        return xu.flatten_nested_tuple(self.parameters_buffers())

    def _compute_loss(self, xla_outputs):
        xla_losses = []
        for _, replica_xla_outputs in enumerate(xla_outputs):
            # The loss is ordinal 0 of the model returned tuple (original model
            # output is ordinal 1).
            xla_losses.append(replica_xla_outputs[0])

        losses = torch_xla._XLAC._xla_to_tensors(xla_losses)
        loss = 0.0
        for closs in losses:
            loss += closs.sum().item()
        return loss / len(losses)

    def train(self, samples_loader, optimizer, batch_size, log_interval=1,
              log_fn=print):
        wloader = LoaderWrapper(samples_loader, self._loader_prefetch,
                                batch_size, num_cores=self._num_cores,
                                devices=self._devices, fused_mode=True)
        wloader_cleaner = xu.Cleaner(wloader.close)
        processed_samples = 0
        loss = None
        start_time = time.time()
        self._epoch += 1
        for batch_number, (inputs, targets) in wloader:
            optimizer.zero_grad()
            xla_outputs = run_xla_model(self._xla_model, inputs,
                                        devices=self._devices)
            xla_run_grad(self._xla_model, self._get_backward_grads(xla_outputs),
                         devices=self._devices)
            optimizer.step()
            processed_samples += self._num_cores * batch_size
            if (log_fn is not None and log_interval is not None and
                    batch_number % log_interval == 0):
                loss = self._compute_loss(xla_outputs)
                log_fn('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                       'Loss: {:.6f}\tSamples/sec: {:.1f}'.format(
                           self._epoch, processed_samples,
                           len(samples_loader) * batch_size,
                           100. * batch_number * self._num_cores /
                           len(samples_loader), loss, processed_samples /
                           (time.time() - start_time)))
        return loss

    def test(self, samples_loader, eval_fn, batch_size, log_fn=print):
        wloader = LoaderWrapper(samples_loader, self._loader_prefetch, batch_size,
                                num_cores=self._num_cores, devices=self._devices,
                                fused_mode=True)
        wloader_cleaner = xu.Cleaner(wloader.close)
        test_loss = 0
        count = 0
        correct = 0
        start_time = time.time()
        with torch.no_grad():
            for batch_number, (inputs, targets) in wloader:
                xla_outputs = run_xla_model(self._xla_model, inputs,
                                            devices=self._devices)
                for i, replica_xla_outputs in enumerate(xla_outputs):
                    # The original model output is ordinal 1 of the returned
                    # tuple (loss is ordinal 0).
                    output = replica_xla_outputs[1].to_tensor()
                    # Inputs [1] is the model target, as inputs with
                    # fused_mode=True are [input, target].
                    closs, ccorrect = eval_fn(output, inputs[i][1].to_tensor())
                    test_loss += closs
                    correct += ccorrect
                    count += batch_size
        test_loss /= count
        accuracy = 100.0 * correct / count
        if log_fn is not None:
            log_fn('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), '
                   'Samples/sec: {:.1f}\n'.format(test_loss, correct, count, accuracy,
                                                  count / (time.time() - start_time)))
        return accuracy
