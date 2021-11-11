import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim

__all__ = ['perfnetB']

class PerfNetB:
    def __init__(self, inputs, training, reg_constant = 0.00001, drop_rate = 0.2, name = ''):
        # init layer
        data_dim = inputs.shape[1]
        out = tf.reshape(inputs, [-1, data_dim, 1])
        
        # layer1 - conv1d
        name_scope = 'layer_conv1' 
        with tf.variable_scope(name_scope) as scope:
            out = slim.conv1d(out, 32, 3, scope='conv1')

        # layer2 - conv1d
        name_scope = 'layer_conv2' 
        with tf.variable_scope(name_scope) as scope:
            fcLayer = slim.conv1d(out, 128, 2, scope='conv2')
            #fcLayer = tf.nn.relu(fcLayer) #acc without relu might be better
        
        fcLayer = slim.flatten(fcLayer)

        # layer3 - fc
        name_scope = name + 'fc1'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = inputs,
                units = 32,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        
        # layer4 - fc
        name_scope = name + 'fc2'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 64,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        
        # layer5 - fc
        name_scope = name + 'fc3'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 128,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)

        # layer6 - fc
        name_scope = name + 'fc4'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 256,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        
        # layer7 - Drop
        name_scope = name + 'Drop1'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dropout(inputs=out,
                rate=drop_rate,
                training=training)
        
        # layer8 - fc (Prediction)
        name_scope = name + 'Prediction'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 1,
                activation = tf.nn.relu,
                use_bias = False)
        
        self.prediction = tf.reshape(out, [-1], name='tf_prediction')

def perfnetB(inputs, training):
    return PerfNetB(inputs, training, name = 'perfnetB')

