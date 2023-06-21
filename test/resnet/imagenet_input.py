# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Efficient ImageNet input pipeline using tf.data.Dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import os
from absl import flags
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS
CROP_PADDING = 32


def get_input_fn(data_dir,
                 is_training,
                 dtype,
                 image_size,
                 reshape_to_r1,
                 shuffle_size=16384,
                 cache_decoded_image=False):
  """Returns the input_fn."""

  def dataset_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.parse_single_example(
        value, {
            'image/encoded': tf.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.FixedLenFeature([], tf.int64, -1)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1

    def preprocess_fn():
      """Preprocess the image."""
      shape = tf.image.extract_jpeg_shape(image_bytes)
      if is_training:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
        bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
            tf.image.extract_jpeg_shape(image_bytes),
            bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(0.75, 1.33),
            area_range=(0.05, 1.0),
            max_attempts=10,
            use_image_if_no_bounding_boxes=True)
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack(
            [offset_y, offset_x, target_height, target_width])
      else:
        crop_size = tf.cast(
            ((image_size / (image_size + CROP_PADDING)) *
             tf.cast(tf.minimum(shape[0], shape[1]), tf.float32)), tf.int32)
        offset_y, offset_x = [
            ((shape[i] - crop_size) + 1) // 2 for i in range(2)
        ]
        crop_window = tf.stack([offset_y, offset_x, crop_size, crop_size])

      image = tf.image.decode_and_crop_jpeg(
          image_bytes, crop_window, channels=3)
      image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
      if is_training:
        image = tf.image.random_flip_left_right(image)
      image = tf.reshape(image, [image_size, image_size, 3])
      return tf.image.convert_image_dtype(image, dtype)

    empty_example = tf.zeros([image_size, image_size, 3], dtype)
    return tf.cond(label < 0, lambda: empty_example, preprocess_fn), label

  def cached_parser(value):
    """Parses an image and its label from a serialized ResNet-50 TFExample."""
    parsed = tf.parse_single_example(
        value, {
            'image/encoded': tf.FixedLenFeature((), tf.string, ''),
            'image/class/label': tf.FixedLenFeature([], tf.int64, -1)
        })
    image_bytes = tf.reshape(parsed['image/encoded'], [])
    image_bytes = tf.io.decode_jpeg(image_bytes, channels=3)
    label = tf.cast(tf.reshape(parsed['image/class/label'], []), tf.int32) - 1
    return image_bytes, label

  def crop_image(image_bytes, label):
    """Preprocess the image."""
    shape = tf.shape(image_bytes)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        shape,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.05, 1.0),
        max_attempts=10,
        use_image_if_no_bounding_boxes=True)
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    image = tf.image.crop_to_bounding_box(image_bytes, offset_y, offset_x,
                                          target_height, target_width)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
    image = tf.image.random_flip_left_right(image)
    image = tf.reshape(image, [image_size, image_size, 3])
    return tf.image.convert_image_dtype(image, dtype), label

  def set_shapes(batch_size, images, labels):
    """Statically set the batch_size dimension."""
    shape = [None, None, batch_size, None]
    images.set_shape(images.get_shape().merge_with(tf.TensorShape(shape)))
    if reshape_to_r1:
      images = tf.reshape(images, [-1])
    labels.set_shape([batch_size])
    return images, labels

  def input_fn(params):
    """Input function which provides a single batch for train or eval."""
    batch_size = params['batch_size']
    index = params['dataset_index']
    num_hosts = params['dataset_num_shards']
    num_dataset_per_shard = max(
        1,
        int(
            math.ceil(50000 / 16384) *
            16384 / 64))
    padded_dataset = tf.data.Dataset.from_tensors(
        tf.constant(
            tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image/class/label':
                            tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[-1])),
                        'image/encoded':
                            tf.train.Feature(
                                bytes_list=tf.train.BytesList(
                                    value=[str.encode('')]))
                    })).SerializeToString(),
            dtype=tf.string)).repeat(num_dataset_per_shard)

    if data_dir is None:
      dataset = padded_dataset.repeat(batch_size * 100)
    else:
      file_pattern = os.path.join(data_dir,
                                  'train-*' if is_training else 'validation-*')
      dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
      dataset = dataset.shard(num_hosts, index)
      dataset = dataset.interleave(tf.data.TFRecordDataset, 64, 1, 64)

      if is_training:
        dataset = dataset.cache().shuffle(shuffle_size).repeat(100)
      else:
        dataset = dataset.concatenate(padded_dataset).take(
            num_dataset_per_shard).repeat(100)

    if cache_decoded_image and is_training:
      dataset = dataset.map(cached_parser,
                            64).repeat().map(crop_image,
                                             64).batch(batch_size, True)
    else:
      dataset = dataset.map(dataset_parser, 64).batch(batch_size, True)

    # Transpose for performance on TPU
    transpose_array = [1, 2, 0, 3]
    dataset = dataset.map(
        lambda imgs, labels: (tf.transpose(imgs, transpose_array), labels), 64)
    dataset = dataset.map(functools.partial(set_shapes, batch_size), 64)
    dataset = dataset.prefetch(64)

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_threading.private_threadpool_size = 100
    dataset = dataset.with_options(options)
    return dataset.as_numpy_iterator()

  return input_fn
