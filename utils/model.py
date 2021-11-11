import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from .network import get_local_nn
from .scheduler import get_scheduler

class Model:
    """Model class"""
    def __init__(self, flags, data_dim):
        ### tf setting
        tf.reset_default_graph()
        self.tf_inputs  = tf.placeholder(tf.float32, shape=(None, data_dim), name='tf_input')
        self.tf_targets = tf.placeholder(tf.float32, shape=(None), name='tf_targets')
        self.tf_lr = tf.placeholder(tf.float32, shape=[], name='tf_learning_rate')
        self.tf_istraining  = tf.placeholder(tf.bool, shape=None, name='tf__istraining')
        self.tf_global_step = tf.get_variable('global_step', initializer=tf.constant(flags.start_epoch), trainable=False)

        ### learning setting
        self.batch_size    = flags.batch_size
        self.loss_name     = flags.loss_name
        #if not flags.evaluate and not flags.load_model :
        self.learning_rate = flags.learning_rate
        self.momentum      = flags.momentum
        self.weight_decay  = flags.weight_decay
        self.pred_device   = flags.predition_device
        self.network_name  = flags.network_name
        ### Adanced params
        self._scheduler = get_scheduler(flags.scheduler_mode, flags.learning_rate, flags.scheduler_from_json,
                flags.scheduler_step, flags.scheduler_gamma)
        self._network = get_local_nn(self.tf_inputs, self.tf_istraining, network_name=flags.network_name)
        self._prediction = None
        self._loss = None
        self._train_op = None
    
    @property
    def scheduler(self):
        return self._scheduler
    
    @property
    def prediction(self):
        if self._prediction is None:
            self._prediction = self._network.prediction
        return self._prediction
    
    @property
    def loss(self):
        if self._loss is None:
            if self.loss_name == "msle":
                self._loss = (tf.losses.get_regularization_loss()
                        + tf.losses.mean_squared_error(
                            labels=tf.log(1+self.tf_targets),
                            predictions=tf.log(1+self.prediction)))
            elif self.loss_name == "mse":
                self._loss = (tf.losses.get_regularization_loss()
                        + tf.losses.mean_squared_error(
                            labels=self.tf_targets,
                            predictions=self.prediction))
            elif self.loss_name == "mae":
                maes = tf.losses.absolute_difference(self.tf_targets, self.prediction)
                self._loss = tf.reduce_mean(maes)
            elif self.loss_name == "maple":
                maes = tf.losses.absolute_difference( tf.log(1+ self.tf_targets), 
                    tf.log(1 + self.prediction))
                self._loss = tf.reduce_mean(maes/tf.log(1+ self.tf_targets))
            elif self.loss_name == "poisson":
                self._loss = tf.keras.losses.poisson(self.tf_targets, tf.exp(self.prediction))
        return self._loss

    @property
    def train_op(self):
        if self._train_op is None:
            opt = tf.train.AdamOptimizer(
                    learning_rate=self.tf_lr,
                    epsilon=.1)
            self._train_op = opt.minimize(
                    self.loss,
                    global_step=self.tf_global_step)
        return self._train_op

    def adjust_lr(self, epoch):
        self.scheduler.adjust_lr(epoch)
        self.learning_rate = self.scheduler.cur_lr