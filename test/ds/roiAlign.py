# Copied from ezyang's python implementation: https://gist.github.com/ezyang/813e86ff5b46ae9e41fc1920790e51fc
import torch
import torch.nn.functional as F
from functorch.dim import dims
import math

import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
pd = torch._C._EnablePythonDispatcher()
xla_dev = xm.xla_device()

# NB: all tensor inputs
def bilinear_interpolate(input, height, width, y, x, ymask, xmask):
    # deal with inverse element out of feature map boundary

    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1. - ly
    hx = 1. - lx

    # do bilinear interpolation, but respect the masking!
    def masked_index(y, x):
        y = torch.where(ymask, y, 0)
        x = torch.where(xmask, x, 0)
        return input[y, x]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx;

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return val

def roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    _, _, height, width = input.size()

    n, c, ph, pw = dims(4)
    ph.size = pooled_height
    pw.size = pooled_width
    offset_rois = rois[n]
    roi_batch_ind = offset_rois[0].int()
    offset = 0.5 if aligned else 0.0
    roi_start_w = offset_rois[1] * spatial_scale - offset
    roi_start_h = offset_rois[2] * spatial_scale - offset
    roi_end_w = offset_rois[3] * spatial_scale - offset
    roi_end_h = offset_rois[4] * spatial_scale - offset

    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)
        roi_height = torch.clamp(roi_height, min=1.0)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    offset_input = input[roi_batch_ind][c]

    roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_height / pooled_height)
    roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_width / pooled_width)

    count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)

    iy, ix = dims(2)
    # OK, so this is a little awkward, in the CUDA kernel, we only loop over
    # the pixels that actually are selected by the region.  We can't easily do
    # that here because we don't have nested tensors.  So we'll just
    # inefficiently loop over everything and mask out things that are not in
    # roi_bin_grid_h
    iy.size = height  # < roi_bin_grid_h
    ix.size = width  # < roi_bin_grid_w
    y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
    x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
    ymask = iy < roi_bin_grid_h
    xmask = ix < roi_bin_grid_w
    val = bilinear_interpolate(offset_input, height, width, y, x, ymask, xmask)
    val = torch.where(ymask, val, 0)
    val = torch.where(xmask, val, 0)
    output = val.sum((iy, ix))
    output /= count

    return output.order(n, c, ph, pw)


# Create a feature map with size (1, 256, 16, 16). 
# This could be an output from a convolutional layer of a CNN
features = torch.randn(1, 256, 16, 16, device=xla_dev)

# Create regions of interest
# Each RoI is defined by a tuple (idx, x1, y1, x2, y2)
# idx is the index into features indicating which image the RoI belongs to
# (x1, y1) and (x2, y2) are the coordinates of the top-left and bottom-right corners of the RoI respectively
rois = torch.tensor([
    [0, 60, 60, 100, 100],
    [0, 120, 120, 160, 160]
], dtype=torch.float, device=xla_dev)

# Set output size and spatial scale
output_size = (7, 7)
spatial_scale = 1.0 / 16.0

# Call the roi_align function
# pooled_features = roi_align(features, rois, spatial_scale, output_size[0], output_size[1], -1, False)
# print(pooled_features.sum())

from torchvision.ops import roi_align as roi_align_torchvision

print(roi_align_torchvision(features, rois, output_size, spatial_scale).sum())
