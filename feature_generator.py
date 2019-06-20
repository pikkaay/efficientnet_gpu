# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Eval checkpoint driver.

This is an example evaluation script for users to understand the EfficientNet
model checkpoints on CPU. To serve EfficientNet, please consider to export a
`SavedModel` from checkpoints and use tf-serving to serve.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf


import efficientnet_builder
import preprocessing


# flags.DEFINE_string('model_name', 'efficientnet-b0', 'Model name to eval.')
# flags.DEFINE_string('runmode', 'examples', 'Running mode: examples or imagenet')
# flags.DEFINE_string('imagenet_eval_glob', None,
#                     'Imagenet eval image glob, '
#                     'such as /imagenet/ILSVRC2012*.JPEG')
# flags.DEFINE_string('imagenet_eval_label', None,
#                     'Imagenet eval label file path, '
#                     'such as /imagenet/ILSVRC2012_validation_ground_truth.txt')
# flags.DEFINE_string('ckpt_dir', '/tmp/ckpt/', 'Checkpoint folders')
# flags.DEFINE_string('example_img', '/tmp/panda.jpg',
#                     'Filepath for a single example image.')
# flags.DEFINE_string('labels_map_file', '/tmp/labels_map.txt',
#                     'Labels map from label id to its meaning.')
# flags.DEFINE_integer('num_images', 5000,
#                      'Number of images to eval. Use -1 to eval all images.')
# FLAGS = flags.FLAGS

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


model_name='efficientnet-b0'
batch_size=256
"""Initialize internal variables."""
model_name = model_name
batch_size = batch_size
num_classes = 1000
# Model Scaling parameters
_, _, image_size, _ = efficientnet_builder.efficientnet_params(
      model_name)

def restore_model(sess, ckpt_dir):
  """Restore variables from checkpoint dir."""
  checkpoint = tf.train.latest_checkpoint(ckpt_dir)
  ema = tf.train.ExponentialMovingAverage(decay=0.9999)
  ema_vars = tf.trainable_variables() + tf.get_collection('moving_vars')
  for v in tf.global_variables():
    if 'moving_mean' in v.name or 'moving_variance' in v.name:
      ema_vars.append(v)
  ema_vars = list(set(ema_vars))
  var_dict = ema.variables_to_restore(ema_vars)
  saver = tf.train.Saver(var_dict, max_to_keep=1)
  saver.restore(sess, checkpoint)

def build_model( features, is_training):
  """Build model with input features."""
  features -= tf.constant(MEAN_RGB, shape=[1, 1, 3], dtype=features.dtype)
  features /= tf.constant(STDDEV_RGB, shape=[1, 1, 3], dtype=features.dtype)
  out, _ = efficientnet_builder.build_model_base(
      features, model_name, is_training)
  return out

def build_dataset( filenames, is_training):
  """Build input dataset."""
  filenames = tf.constant(filenames)
  dataset = tf.data.Dataset.from_tensor_slices((filenames))

  def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = preprocessing.preprocess_image(
        image_string, is_training, image_size=image_size)
    image = tf.cast(image_decoded, tf.float32)
    return image

  dataset = dataset.map(_parse_function)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images = iterator.get_next()
  return images

import pandas as pd
import numpy as np

path = './../../../../main/'
# path = './'
train = pd.read_csv(path+'./CheXpert-v1.0-small/train.csv')
valid = pd.read_csv(path+'./CheXpert-v1.0-small/valid.csv')

train['validation'] = False
valid['validation'] = True
df = pd.concat([train, valid])

columns = ['Path', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'validation']
df = df[columns]

for feature in ['Atelectasis', 'Edema']:
    df[feature] = df[feature].apply(lambda x: 1 if x==-1 else x)

for feature in ['Cardiomegaly', 'Consolidation', 'Pleural Effusion']:
    df[feature] = df[feature].apply(lambda x: 0 if x==-1 else x)
df.fillna(0, inplace=True)

train = df[~df.validation]
print(len(train))
training_files = train['Path'].tolist()
training_files = [path+fil for fil in training_files]

def main():
  graph = tf.Graph()
  with tf.Session(graph=graph) as sess:
    images = build_dataset(training_files, False)
    out = build_model(images, is_training=False)
    out = tf.reduce_mean(out, axis=[1,2])
    sess.run(tf.global_variables_initializer())
    restore_model(sess, './weights_efficientnet-b0/')
    out_probs = []
    for i in range(len(training_files) // batch_size):
      out_probs.append(sess.run(out))
      print(i)
  np.save('file.npy', np.array(out_probs))  

if __name__ == '__main__':
  main()