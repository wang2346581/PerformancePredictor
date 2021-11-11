import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim

__all__ = ['perfnetAS']

class PerfNetAS:
    def __init__(self, inputs, training, reg_constant = 0.00001, drop_rate = 0.2, name = ''):
        # layer1 - fc
        name_scope = name + 'fc1'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = inputs,
                units = 256,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        
        # layer2 - fc
        name_scope = name + 'fc2'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 256,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        
        # layer3 - fc
        name_scope = name + 'fc3'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 256,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)

        # layer4 - fc
        name_scope = name + 'fc4'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 256,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        
        # layer5 - Drop
        name_scope = name + 'Drop1'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dropout(inputs=out,
                rate=drop_rate,
                training=training)
        
        # layer6 - fc (Prediction)
        name_scope = name + 'Prediction'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 1,
                activation = tf.nn.relu,
                use_bias = False)
        
        self.prediction = tf.reshape(out, [-1], name='tf_prediction')

def perfnetAS(inputs, training):
    return PerfNetAS(inputs, training, name = 'perfnetAS')
