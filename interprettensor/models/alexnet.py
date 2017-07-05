# Copyright 2017 Ruth Fong. All Rights Reserved.
#
# ==============================================================================
"""Contains a model definition for AlexNet using LRP layers. 

Code based on the following:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/nets/alexnet.py
https://github.com/guerzh/tf_weights/blob/master/myalexnet_forward_newtf.py

Pre-trained weights can be downloaded from the below:
http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy

Note: This is based on tf_weights, which doesn't work perfectly / seems to do well on some images but not others, 
as noted at https://github.com/guerzh/tf_weights/issues/.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope

from interprettensor.modules.sequential import Sequential
from interprettensor.modules.linear import Linear
from interprettensor.modules.convolution import Convolution
from interprettensor.modules.maxpool import MaxPool

import numpy as np

def alexnet(inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='alexnet',
               pretrained_weights=None):
  """AlexNet.
  
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 227x227. To use in fully
        convolutional mode, set spatial_squeeze to false.
        The LRN layers have been removed and change the initializers from
        random_normal_initializer to xavier_initializer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    pretrained_weights: path to the .npy weights file that can be downloaded from  
      http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
  Returns:
    the last op containing the log predictions and end_points dict.
  """

  batch_size = inputs.shape[0].value
  input_dim = inputs.shape[1].value
  assert(input_dim == inputs.shape[2].value)
  input_depth = inputs.shape[3].value

  with variable_scope.variable_scope(scope, 'alexnet', [inputs]) as sc:
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    keep_prob = dropout_keep_prob if is_training else 1.0
    if pretrained_weights is None:
        weights = {
                'conv1':None,
                'conv2':None,
                'conv3':None,
                'conv4':None,
                'conv5':None,
                'fc6':None,
                'fc7':None,
                'fc8':None,
        }
        biases = weights
    else:
        print('Loading weights from %s ...' % pretrained_weights)
        pretrained_weights = np.load(pretrained_weights).item()
        weights = {k:v[0] for k,v in pretrained_weights.items()}
        biases = {k:v[1] for k,v in pretrained_weights.items()}
                  
    layers = [
          Convolution(output_depth=96, batch_size=batch_size, input_dim=input_dim, input_depth=input_depth,
            kernel_size=11, stride_size=4, 
            act='relu', pad='VALID', weights=weights['conv1'], biases=biases['conv1'], name='conv1'),
          MaxPool(pool_size=3, pool_stride=[1,2,2,1], pad='VALID', name='pool1'),
          Convolution(output_depth=256, groups=2, kernel_size=5, stride_size=1, act='relu', pad='SAME', 
              weights=weights['conv2'], biases=biases['conv2'], name='conv2'),
          MaxPool(pool_size=3, pool_stride=[1,2,2,1], pad='VALID', name='pool2'),
          Convolution(output_depth=384, groups=1, kernel_size=3, stride_size=1, act='relu', pad='SAME', 
              weights=weights['conv3'], biases=biases['conv3'], name='conv3'),
          Convolution(output_depth=384, groups=2, kernel_size=3, stride_size=1, act='relu', pad='SAME', 
              weights=weights['conv4'], biases=biases['conv4'], name='conv4'),
          Convolution(output_depth=256, groups=2,kernel_size=3, stride_size=1, act='relu', pad='SAME', 
              weights=weights['conv5'], biases=biases['conv5'], name='conv5'),
          MaxPool(pool_size=3, pool_stride=[1,2,2,1], pad='VALID', name='pool5'),
          Convolution(output_depth=4096, kernel_size=6, stride_size=1, keep_prob=keep_prob, 
            act='relu', pad='VALID', weights=weights['fc6'], biases=biases['fc6'], name='fc6'),
          Convolution(output_depth=4096, kernel_size=1, stride_size=1, keep_prob=keep_prob, 
            act='relu', pad='SAME', weights=weights['fc7'], biases=biases['fc7'], name='fc7'),
          Convolution(output_depth=num_classes, kernel_size=1, stride_size=1, 
            act='linear', pad='SAME', weights=weights['fc8'], biases=biases['fc8'], name='fc8'),
    ]

    net = Sequential(layers)
    out = net.forward(inputs)

    end_points = {}
    for l in layers:
        end_points[l.name.split('_')[0]] = l

    if spatial_squeeze:
        out = array_ops.squeeze(out, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = out 

    return out, end_points

alexnet.default_image_size = 227 
