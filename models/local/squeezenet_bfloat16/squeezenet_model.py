# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""SqueezeNet implementation with TPU support.

This version does not contain the model compression components (
sparsification and quantization).

Original paper: (https://arxiv.org/pdf/1602.07360.pdf)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer


def get_custom_getter():
  """Returns a custom getter that this class's methods must be called under.

  All methods of this class must be called under a variable scope that was
  passed this custom getter. Example:

  ```python
  network = ConvNetBuilder(...)
  with tf.variable_scope('cg', custom_getter=network.get_custom_getter()):
    network.conv(...)
    # Call more methods of network here
  ```

  Currently, this custom getter only does anything if self.use_tf_layers is
  True. In that case, it causes variables to be stored as dtype
  self.variable_type, then casted to the requested dtype, instead of directly
  storing the variable as the requested dtype.
  """

  def inner_custom_getter(getter, *args, **kwargs):
    """Custom getter that forces variables to have type self.variable_type."""
    cast_to_bfloat16 = False
    requested_dtype = kwargs['dtype']
    if requested_dtype == tf.bfloat16:
      # Only change the variable dtype if doing so does not decrease variable
      # precision.
      kwargs['dtype'] = tf.float32
      cast_to_bfloat16 = True
    var = getter(*args, **kwargs)
    # This if statement is needed to guard the cast, because batch norm
    # assigns directly to the return value of this custom getter. The cast
    # makes the return value not a variable so it cannot be assigned. Batch
    # norm variables are always in fp32 so this if statement is never
    # triggered for them.
    if cast_to_bfloat16:
      var = tf.cast(var, tf.bfloat16)
      print('to bfloat16', var)
    return var

  return inner_custom_getter

def conv2d(inputs,
           filters,
           kernel_size,
           strides=(1, 1),
           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
           bias_initializer=tf.zeros_initializer(),
           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
           name=None):
  return tf.layers.conv2d(
      inputs,
      filters,
      kernel_size,
      strides,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      activation=tf.nn.relu,
      name=name,
      padding="same")


def fire_module(inputs, squeeze_depth, expand_depth, name):
  """Fire module: squeeze input filters, then apply spatial convolutions."""
  with tf.variable_scope(name, "fire", [inputs]):
    squeezed = conv2d(inputs, squeeze_depth, [1, 1], name="squeeze")
    e1x1 = conv2d(squeezed, expand_depth, [1, 1], name="e1x1")
    e3x3 = conv2d(squeezed, expand_depth, [3, 3], name="e3x3")
    return tf.concat([e1x1, e3x3], axis=3)


def squeezenet(images, is_training=True, num_classes=1001):
  """Squeezenet 1.0 model."""
  net = conv2d(images, 96, [7, 7], strides=(2, 2), name="conv1")
  net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool1")
  net = fire_module(net, 16, 64, name="fire2")
  net = fire_module(net, 16, 64, name="fire3")
  net = fire_module(net, 32, 128, name="fire4")
  net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool4")
  net = fire_module(net, 32, 128, name="fire5")
  net = fire_module(net, 48, 192, name="fire6")
  net = fire_module(net, 48, 192, name="fire7")
  net = fire_module(net, 64, 256, name="fire8")
  net = tf.layers.max_pooling2d(net, [3, 3], strides=(2, 2), name="maxpool8")
  net = fire_module(net, 64, 256, name="fire9")
  net = tf.layers.dropout(net, rate=0.5 if is_training else 0.0, name="drop9")
  net = conv2d(net, num_classes, [1, 1], strides=(1, 1), name="conv10")
  net = tf.layers.average_pooling2d(net, pool_size=(13, 13), strides=(1, 1))
  logits = tf.layers.flatten(net)
  return logits


def metric_fn(labels, logits, learning_rate):
  predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
  labels = tf.cast(labels, tf.int64)
  return {
      "accuracy": tf.metrics.accuracy(labels, predictions),
      "recall_at_5": tf.metrics.recall_at_k(labels, logits, 5),
      "recall_at_1": tf.metrics.recall_at_k(labels, logits, 1),
      "learning_rate": tf.metrics.mean(learning_rate),
  }

def model_fn(features, labels, mode, params):
  """TPUEstimatorSpec for the Squeezenet model."""
  ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
  is_training = mode == tf.estimator.ModeKeys.TRAIN
  with tf.variable_scope('cg', custom_getter=get_custom_getter()):
    logits = squeezenet(
      features, is_training=is_training, num_classes=params["num_classes"])
    logits = tf.cast(logits, tf.float32)

  loss = tf.reduce_mean(
      tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels))

  global_batch_size = params["num_shards"] * params["batch_size"]
  decay_steps = 1300 * 1000 * params["num_epochs"] // global_batch_size
  learning_rate = tf.train.polynomial_decay(
      params["lr"],
      global_step=tf.train.get_or_create_global_step(),
      end_learning_rate=params["min_lr"],
      decay_steps=decay_steps,
      power=1.0,
      cycle=False)

  # TODO(power): Hack copied from resnet: remove when summaries are working.
  lr_repeat = tf.reshape(
      tf.tile(tf.expand_dims(learning_rate, 0), [params["batch_size"],]),
      [params["batch_size"], 1])

  if params["optimizer"] == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  elif params["optimizer"] == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=learning_rate,
        momentum=params["momentum"],
        epsilon=1.0
    )
  else:
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=params["momentum"],
        use_nesterov=True)

  if params["use_tpu"]:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

  train_op = optimizer.minimize(loss, tf.train.get_global_step())

  param_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=ProfileOptionBuilder.trainable_variables_parameter())
  fl_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())

  return tpu_estimator.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metrics=(metric_fn, [labels, logits, lr_repeat]),
      predictions={
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      },
  )
