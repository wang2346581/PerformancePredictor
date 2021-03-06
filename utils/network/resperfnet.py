import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim

__all__ = ['resperfnet']

class ResPerfNet:
    def __init__(self, inputs, training, reg_constant = 0.00001, drop_rate = 0.2, name = ''):#dropout = 0.2 ##ori ratio
        # init layer
        data_dim = inputs.shape[1]
        out = tf.reshape(inputs, [-1, data_dim, 1])
        
        # layer1 - conv1d
        name_scope = 'reslayer_conv1' 
        with tf.variable_scope(name_scope) as _:
            out = slim.conv1d(out, 128, 3, scope='conv1')

        name_scope = 'reslayer_conv2' 
        with tf.variable_scope(name_scope) as scope:
            out1 = slim.conv1d(out, 128, 1, scope='conv2-1')
            out1 = tf.nn.relu(out1)
            out1 = slim.conv1d(out1, 128, 1, scope='conv2-2')
            out1 = tf.nn.relu(out1)
            out = tf.add(out, out1)
            out = tf.nn.relu(out)

        name_scope = 'reslayer_conv3' 
        with tf.variable_scope(name_scope) as scope:
            out1 = slim.conv1d(out, 128, 1, scope='conv3-1')
            out1 = tf.nn.relu(out1)
            out1 = slim.conv1d(out1, 128, 1, scope='conv3-2')
            out1 = tf.nn.relu(out1)
            out = tf.add(out, out1)
            out = tf.nn.relu(out)
#        fcLayer = slim.flatten(out)
        
        name_scope = 'reslayer_tmp1'
        with tf.variable_scope(name_scope) as scope:
            out = slim.conv1d(out, 64, 3, scope='convtmp1')

#       out = tf.layers.average_pooling1d(net, pool_size=2, strides=2, padding='same')

        name_scope = 'reslayer_conv4' 
        with tf.variable_scope(name_scope) as scope:
            out1 = slim.conv1d(out, 64, 1, scope='conv4-1')
            out1 = tf.nn.relu(out1)
            out1 = slim.conv1d(out1, 64, 1, scope='conv4-2')
            out1 = tf.nn.relu(out1)
            out = tf.add(out, out1)
            out = tf.nn.relu(out)


        name_scope = 'reslayer_conv5' 
        with tf.variable_scope(name_scope) as scope:
            out1 = slim.conv1d(out, 64, 1, scope='conv5-1')
            out1 = tf.nn.relu(out1)
            out1 = slim.conv1d(out1, 64, 1, scope='conv5-2')
            out1 = tf.nn.relu(out1)
            out = tf.add(out, out1)
            out = tf.nn.relu(out)

        name_scope = 'reslayer_tmp2'
        with tf.variable_scope(name_scope) as scope:
            out = slim.conv1d(out, 32, 2, scope='convtmp2')

#        out = tf.layers.average_pooling1d(net, pool_size=2, strides=2, padding='same')

        
        name_scope = 'reslayer_conv6' 
        with tf.variable_scope(name_scope) as scope:
            out1 = slim.conv1d(out, 32, 1, scope='conv6-1')
            out1 = tf.nn.relu(out1)
            out1 = slim.conv1d(out1, 32, 1, scope='conv6-2')
            out1 = tf.nn.relu(out1)
            out = tf.add(out, out1)
            out = tf.nn.relu(out)


        name_scope = 'reslayer_conv7' 
        with tf.variable_scope(name_scope) as scope:
            out1 = slim.conv1d(out, 32, 1, scope='conv7-1')
            out1 = tf.nn.relu(out1)
            out1 = slim.conv1d(out1, 32, 1, scope='conv7-2')
            out1 = tf.nn.relu(out1)
            out = tf.add(out, out1)
            out = tf.nn.relu(out)


        fcLayer = slim.flatten(out)

        
        # layer3 - fc
        name_scope = name + 'fc1'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = fcLayer,
                units = 128,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        
        # layer4 - fc
        name_scope = name + 'fc2'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 128, #64,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
         
        # layer5 - fc
        name_scope = name + 'fc3'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 128, #128,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        '''
        # layer6 - fc
        name_scope = name + 'fc4'
        with tf.variable_scope(name_scope) as _:
            out = tf.layers.dense(
                inputs = out,
                units = 256,
                activation = tf.nn.relu,
                kernel_regularizer = tf.contrib.layers.l2_regularizer(reg_constant),
                use_bias = True)
        '''
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

def resperfnet(inputs, training):
    return ResPerfNet(inputs, training, name = 'resperfnet')

