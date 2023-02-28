import argparse
import os
import sys
import math

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--verbosity', type=int, default=2)
FLAGS, leftovers = parser.parse_known_args()
sys.argv = [sys.argv[0]] + leftovers

import math
import numpy as np
import unittest
import torch
import torchvision
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torchvision
from torch.autograd import gradcheck

# It enables us to run python implementations of CompositeAutogradImplicit ops.
# CompositeAutogradImplicit means we don't have an explicit backward formula for an op instead an op is composed of a bunch of ops that do have backward formulas and combines this formulas is equivalent to differentiating the op explicitly.
pd = torch._C._EnablePythonDispatcher()
xla_dev = xm.xla_device()


class Feedforward(torch.nn.Module):

  def __init__(self, input_size, hidden_size):
    super().__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
    self.fc1.weight.data.fill_(0.01)
    self.fc1.bias.data.fill_(0.01)
    self.relu = torch.nn.ReLU()
    self.fc2 = torch.nn.Linear(self.hidden_size, 1)
    self.fc2.weight.data.fill_(0.01)
    self.fc2.bias.data.fill_(0.01)
    self.sigmoid = torch.nn.Sigmoid()

  def forward(self, x):
    hidden = self.fc1(x)
    relu = self.relu(hidden)
    output = self.fc2(relu)
    output = self.sigmoid(output)
    return output


@unittest.skipIf(
    not xm.get_xla_supported_devices("GPU") and
    not xm.get_xla_supported_devices("TPU"),
    f"The tests fail on CPU. See https://github.com/pytorch/xla/issues/4298 for more detail."
)
class TestDynamicShapeModels(unittest.TestCase):

  @unittest.skip("Broke by functionalization")
  def test_forward_pass_dynamic_input_correctness(self):
    losses = []
    for _ in range(2):
      num_features = 2
      num_test_samples = 5
      x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                     num_features, xla_dev)

      model = Feedforward(num_features, hidden_size=10).to(xla_dev)
      criterion = torch.nn.BCELoss()

      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        before_train = criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
        losses.append(before_train.item())

    np.testing.assert_allclose(losses[0], losses[1], rtol=1e-2, atol=1e-2)
    print('Test passed.')

  @unittest.skip("Broke by functionalization")
  def test_forward_pass_dynamic_input_compile_once(self):
    met.clear_metrics()
    num_compilation_recorded = False
    num_compilation = -1
    for i in range(10):
      num_features = 2
      num_test_samples = 5
      x_test, y_test = self.create_dynamic_test_data(
          num_test_samples, num_features, xla_dev, num_non_zeros=i)

      model = Feedforward(num_features, hidden_size=10).to(xla_dev)
      criterion = torch.nn.BCELoss()

      model.eval()
      with torch.no_grad():
        y_pred = model(x_test)
        criterion(y_pred.squeeze(), y_test)
        xm.mark_step()
        if not num_compilation_recorded:
          num_compilation = met.metric_data('CompileTime')[0]
          num_compilation_recorded = True
        else:
          self.assertEqual(num_compilation,
                           met.metric_data('CompileTime')[0],
                           'number of compilation should not increase.')

  def test_backward_pass_with_dynamic_input(self):
    num_features = 2
    num_test_samples = 5
    model = Feedforward(num_features, hidden_size=10).to(xla_dev)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # training
    model.train()
    x_training, y_training = self.create_dynamic_test_data(
        num_test_samples, num_features, xla_dev)
    y_pred = model(x_training)
    loss = criterion(y_pred.squeeze(), y_training)
    # Backpropagation.
    loss.backward()
    xm.optimizer_step(optimizer)
    print('Finished training.')

    # testing
    model.eval()
    with torch.no_grad():
      x_test, y_test = self.create_dynamic_test_data(num_test_samples,
                                                     num_features, xla_dev)
      y_pred = model(x_test)
      criterion(y_pred.squeeze(), y_test).item()
      xm.mark_step()
    print('Test passed.')

  def test_roialign_forward(self):
    device = xla_dev
    aligned = True
    # contiguous = True
    dtype = torch.float64
    x_dtype, rois_dtype = dtype, dtype
    pool_size = 5
    # n_channels % (pool_size ** 2) == 0 required for PS operations.
    n_channels = 2 * (pool_size**2)
    x = torch.rand(2, n_channels, 10, 10, dtype=x_dtype, device=device)
    rois = torch.tensor(
        [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9], [1, 0, 0, 9, 9]],  # format is (xyxy)
        dtype=rois_dtype,
        device=device,
    )    

    pool_h, pool_w = pool_size, pool_size
    spatial_scale, sampling_ratio=1, -1
    y = torchvision.ops.RoIAlign((pool_h, pool_w), spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned)(x, rois)

    def expected_fn(
        in_data,
        rois,
        pool_h,
        pool_w,
        spatial_scale=1,
        sampling_ratio=-1,
        aligned=False,
        device=None,
        dtype=torch.float64,
    ):
        print('xw32 expected_fn aligned=', aligned)
        if device is None:
            device = torch.device("cpu")
        n_channels = in_data.size(1)
        out_data = torch.zeros(rois.size(0), n_channels, pool_h, pool_w, dtype=dtype, device=device)

        offset = 0.5 if aligned else 0.0

        for r, roi in enumerate(rois):
            batch_idx = int(roi[0])
            j_begin, i_begin, j_end, i_end = (x.item() * spatial_scale - offset for x in roi[1:])

            roi_h = i_end - i_begin
            roi_w = j_end - j_begin
            bin_h = roi_h / pool_h
            bin_w = roi_w / pool_w

            for i in range(0, pool_h):
                start_h = i_begin + i * bin_h
                grid_h = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_h))
                for j in range(0, pool_w):
                    start_w = j_begin + j * bin_w
                    grid_w = sampling_ratio if sampling_ratio > 0 else int(np.ceil(bin_w))

                    for channel in range(0, n_channels):

                        val = 0
                        for iy in range(0, grid_h):
                            y = start_h + (iy + 0.5) * bin_h / grid_h
                            for ix in range(0, grid_w):
                                x = start_w + (ix + 0.5) * bin_w / grid_w
                                val += bilinear_interpolate(in_data[batch_idx, channel, :, :], y, x, snap_border=True)
                        val /= grid_h * grid_w

                        out_data[r, channel, i, j] = val
        return out_data
    y_expected = expected_fn(x, rois, pool_h, pool_w, spatial_scale=1, sampling_ratio=-1, device=device, dtype=x_dtype, aligned=aligned)
    tol = 1e-3 if (x_dtype is torch.half or rois_dtype is torch.half) else 1e-5
    torch.testing.assert_close(y_expected.to(y), y, rtol=tol, atol=tol)
    print('test passes')
    
  def test_roialign_backward(self):
    seed = 1
    device = 'cpu'
    contiguous = True
    pool_size = 2
    dtype = torch.float64
    x = torch.rand(1, 2 * (pool_size**2), 5, 5, dtype=dtype, device=device, requires_grad=True)
    rois = torch.tensor(
       [[0, 0, 0, 4, 4], [0, 0, 2, 3, 4], [0, 2, 2, 4, 4]], dtype=dtype, device=device  # format is (xyxy)
    )

    def func(z):
      return torchvision.ops.RoIAlign((pool_size, pool_size), spatial_scale=1, sampling_ratio=-1, aligned=False)(z, rois)
    def script_func(x):
      scripted = torch.jit.script(torchvision.ops.roi_align)
      return scripted(x, rois, pool_size)
    
    gradcheck(func, (x,))
    gradcheck(script_func, (x,))

     


  def create_dynamic_test_data(self,
                               num_test_samples,
                               num_features,
                               device,
                               num_non_zeros=1):
    x_test = torch.zeros(num_test_samples, num_features)
    num_non_zero_added = 0
    for i in range(num_test_samples):
      for j in range(num_features):
        x_test[i][j] = 1
        num_non_zero_added += 1
        if num_non_zero_added == num_non_zeros:
          break
      if num_non_zero_added == num_non_zeros:
        break

    num_non_zero_added = 0
    y_test = torch.zeros(num_test_samples * 2)
    for i in range(num_test_samples * 2):
      y_test[i] = 1
      num_non_zero_added += 1
      if num_non_zero_added == num_non_zeros:
        break

    x_test_xla = x_test.to(device)
    x_test_nonzero_dev = torch.nonzero(x_test_xla.int()).float()
    y_test_xla = y_test.to(device)
    y_test_nonzero_dev = torch.nonzero(y_test_xla.int()).float().squeeze()
    return x_test_nonzero_dev, y_test_nonzero_dev
  
  def test_roialign_forward(self):
    device = xla_dev
    aligned = True
    # contiguous = True
    dtype = torch.float64
    x_dtype, rois_dtype = dtype, dtype
    pool_size = 5
    # n_channels % (pool_size ** 2) == 0 required for PS operations.
    n_channels = 2 * (pool_size**2)
    x = torch.rand(2, n_channels, 10, 10, dtype=x_dtype, device=device)
    rois = torch.tensor(
        [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9]
        ],  # format is (xyxy)
        dtype=rois_dtype,
        device=device,
    )

    pool_h, pool_w = pool_size, pool_size
    spatial_scale, sampling_ratio = 1, -1
    y = torchvision.ops.RoIAlign((pool_h, pool_w),
                                 spatial_scale=spatial_scale,
                                 sampling_ratio=sampling_ratio,
                                 aligned=aligned).to(device)(x, rois)
    xm.mark_step()

    x_aten = torch.rand(2, n_channels, 10, 10, dtype=x_dtype, device='cpu')
    rois_aten = torch.tensor(
        [[0, 0, 0, 9, 9], [0, 0, 5, 4, 9], [0, 5, 5, 9, 9]
        ],  # format is (xyxy)
        dtype=rois_dtype,
        device='cpu',
    )
    y_aten = torchvision.ops.RoIAlign((pool_h, pool_w),
                                 spatial_scale=spatial_scale,
                                 sampling_ratio=sampling_ratio,
                                 aligned=aligned)(x_aten, rois_aten)
    tol = 1e-3 if (x_dtype is torch.half or rois_dtype is torch.half) else 1e-5
    torch.testing.assert_close(y.to(y_aten), y_aten, rtol=tol, atol=tol)
    print('test passes')
    print(met.metrics_report())

  def test_roialign_backward(self):
    seed = 1
    #device = 'cpu'
    device = xla_dev
    contiguous = True
    pool_size = 2
    dtype = torch.float64
    x = torch.rand(
        1,
        2 * (pool_size**2),
        5,
        5,
        dtype=dtype,
        device=device,
        requires_grad=True)
    rois = torch.tensor(
        [[0, 0, 0, 4, 4], [0, 0, 2, 3, 4], [0, 2, 2, 4, 4]],
        dtype=dtype,
        device=device  # format is (xyxy)
    )

    def func(z):
      return torchvision.ops.RoIAlign((pool_size, pool_size),
                                      spatial_scale=1,
                                      sampling_ratio=-1,
                                      aligned=False).to(device)(z, rois)

    def script_func(x):
      scripted = torch.jit.script(torchvision.ops.roi_align)
      return scripted(x, rois, pool_size)

    gradcheck(func, (x,))
    gradcheck(script_func, (x,))
    print('test passes')
    print(met.metrics_report())

def bilinear_interpolate(data, y, x, snap_border=False):
  height, width = data.shape

  if snap_border:
    if -1 < y <= 0:
      y = 0
    elif height - 1 <= y < height:
      y = height - 1

    if -1 < x <= 0:
      x = 0
    elif width - 1 <= x < width:
      x = width - 1

  y_low = int(math.floor(y))
  x_low = int(math.floor(x))
  y_high = y_low + 1
  x_high = x_low + 1

  wy_h = y - y_low
  wx_h = x - x_low
  wy_l = 1 - wy_h
  wx_l = 1 - wx_h

  val = 0
  for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
    for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
      if 0 <= yp < height and 0 <= xp < width:
        val += wx * wy * data[yp, xp]
  return val

def bilinear_interpolate(data, y, x, snap_border=False):
    height, width = data.shape

    if snap_border:
        if -1 < y <= 0:
            y = 0
        elif height - 1 <= y < height:
            y = height - 1

        if -1 < x <= 0:
            x = 0
        elif width - 1 <= x < width:
            x = width - 1

    y_low = int(math.floor(y))
    x_low = int(math.floor(x))
    y_high = y_low + 1
    x_high = x_low + 1

    wy_h = y - y_low
    wx_h = x - x_low
    wy_l = 1 - wy_h
    wx_l = 1 - wx_h

    val = 0
    for wx, xp in zip((wx_l, wx_h), (x_low, x_high)):
        for wy, yp in zip((wy_l, wy_h), (y_low, y_high)):
            if 0 <= yp < height and 0 <= xp < width:
                val += wx * wy * data[yp, xp]
    return val

if __name__ == '__main__':
  assert os.environ['XLA_EXPERIMENTAL'] != ''
  test = unittest.main(verbosity=FLAGS.verbosity, exit=False)
  # DISABLE PYTHON DISPATCHER FLAG
  del pd
  sys.exit(0 if test.result.wasSuccessful() else 1)
