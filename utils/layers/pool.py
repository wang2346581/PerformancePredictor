import numpy as np
from utils.utils import get_pool_colnames, get_support_activation
from utils.layers.runner import Runner, RunnerRT
from utils.layers.train import TrainCore

__all__ = ['Pooling', 'PoolingInf', 'PoolingTrain', 'PoolingRT']

class PoolingBase():
    def __init__(self, batchsize, matsize,
        channels_in, poolsize, strides, padding, 
        activation_fct, elements_matrix, **kwargs):
        self._batchsize      = batchsize
        self._matsize        = matsize
        self._channels_in    = channels_in
        self._poolsize       = poolsize
        self._strides        = strides
        self._padding        = padding
        self._activation_fct = activation_fct
        self._elements_matrix = elements_matrix
        super(PoolingBase, self).__init__(**kwargs)
    
    def __str__(self):
        str_ = ""
        for i in get_pool_colnames():
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        return str_

    @property
    def batchsize(self):
        return self._batchsize

    @property
    def matsize(self):
        return self._matsize
    
    @property
    def channels_in(self):
        return self._channels_in

    @property
    def poolsize(self):
        return self._poolsize

    @property
    def strides(self):
        return self._strides
    
    @property
    def padding(self):
        return self._padding

    @property
    def activation_fct(self):
        return self._activation_fct

    @property
    def elements_matrix(self):
        return self._elements_matrix

class PoolingInf(PoolingBase, Runner):
    def __init__(self, memcopyin, output_path, 
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, matsize, channels_in, 
        poolsize, strides, padding, 
        activation_fct, elements_matrix, **kwargs):
        super(PoolingInf, self).__init__(memcopyin=memcopyin, output_path=output_path, 
            iter_warmup=iter_warmup, iter_benchmark=iter_benchmark, 
            hashkey=hashkey, batchsize=batchsize, matsize=matsize, 
            channels_in=channels_in, poolsize=poolsize, 
            strides=strides, padding=padding, 
            activation_fct=activation_fct, elements_matrix=elements_matrix, **kwargs)
    
    def gen_input(self):
        import tensorflow as tf
        if not self._input:
            if self.memcopyin:
                self._input = tf.placeholder(tf.float32, shape=[None, self.matsize, 
                    self.matsize, self.channels_in], name=self.tensor_input_name)
            else:
                self._input = tf.Variable(tf.random_normal([self.batchsize, 
                    self.matsize, self.matsize, self.channels_in]))
    
    def gen_np_input(self):
        self.np_input = np.random.normal(127, 60, (self.batchsize, self.matsize, self.matsize, self.channels_in)).astype(float)
    
    def gen_op(self):
        import tensorflow as tf
        self.gen_input()
        self._op = tf.layers.max_pooling2d(self.input, 
            pool_size=(self.poolsize, self.poolsize),
            strides=(self.strides, self.strides), 
            padding=('SAME' if self.padding == 1 else 'VALID'))

    def __str__(self):
        str_ = ""
        for i in get_pool_colnames():
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        str_ += "hashkey: " + str(self.hashkey) + ", "
        str_ += "output_path: " + str(self.output_path)
        return str_   

class PoolingTrain(PoolingInf, TrainCore):
    def __init__(self, memcopyin, output_path, 
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, matsize, channels_in, 
        poolsize, strides, padding, 
        activation_fct, elements_matrix, 
        sgd, rmsprop, adagrad, adam):

        super(PoolingTrain, self).__init__(memcopyin=memcopyin, 
            output_path = output_path, 
            iter_warmup = iter_warmup, iter_benchmark = iter_benchmark, hashkey = hashkey,
            batchsize=batchsize, matsize=matsize,
            channels_in=channels_in, poolsize=poolsize,
            strides=strides, padding=padding, 
            activation_fct=activation_fct, elements_matrix=elements_matrix,
            sgd=sgd, rmsprop=rmsprop, adagrad=adagrad, adam=adam)
    
    def gen_output(self):
        import tensorflow as tf
        if self.padding == 1:
            target_size = np.ceil(np.float(self.matsize/self.strides))
        else:
            target_size = np.ceil(np.float((self.matsize-(self.poolsize-1))/self.strides))
        self.target = tf.Variable(tf.ones([
                    self.batchsize,
                    int(target_size),
                    int(target_size),
                    self.channels_in],
                    dtype=float))
    
    def gen_op(self):
        import tensorflow as tf
        super(PoolingTrain, self).gen_op()
        self.gen_output()
        self.gen_opt()
        loss_ = tf.reduce_mean(tf.square(self.op - self.target))
        self.op = self.opt.minimize(loss=loss_)
    
    def __str__(self):
        str_ = ""
        for i in get_pool_colnames(1):
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        str_ += "hashkey: " + str(self.hashkey) + ", "
        str_ += "output_path: " + str(self.output_path)
        return str_

class PoolingRT(PoolingBase, RunnerRT):
    def __init__(self, memcopyin, output_path, 
        output_pb_path, output_json_path,
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, matsize, channels_in, 
        poolsize, strides, padding, 
        activation_fct, elements_matrix):

        super(PoolingRT, self).__init__(memcopyin=memcopyin, 
            output_path = output_path, 
            output_pb_path = output_pb_path, output_json_path = output_json_path,
            iter_warmup = iter_warmup, iter_benchmark = iter_benchmark, 
            hashkey=hashkey, batchsize=batchsize, matsize=matsize, 
            channels_in=channels_in, poolsize=poolsize, 
            strides=strides, padding=padding, 
            activation_fct=activation_fct, elements_matrix=elements_matrix)
    
    def get_batchsize(self):
        return self.batchsize
    
    def gen_input(self):
        import tensorflow as tf
        if not self._input:
            if self.memcopyin:
                self._input = tf.placeholder(tf.float32, shape=[None, self.matsize, 
                    self.matsize, self.channels_in], name=self.tensor_input_name)
            else:
                self._input = tf.Variable(tf.random_normal([self.batchsize, 
                    self.matsize, self.matsize, self.channels_in]))
    
    def gen_np_input(self):
        self.np_input = np.random.normal(127, 60, (self.batchsize, self.matsize, self.matsize, self.channels_in)).astype(float)
    
    def gen_op(self):
        import tensorflow as tf
        self.gen_input()
        op1 = tf.layers.max_pooling2d(self.input, 
            pool_size=(self.poolsize, self.poolsize),
            strides=(self.strides, self.strides), 
            padding=('SAME' if self.padding == 1 else 'VALID'))
            
        op2 = tf.layers.max_pooling2d(self.input, 
            pool_size=(self.poolsize, self.poolsize),
            strides=(self.strides, self.strides), 
            padding=('SAME' if self.padding == 1 else 'VALID'))
        op1 = tf.contrib.layers.flatten(op1)
        op2 = tf.contrib.layers.flatten(op2) 
        self.op  = tf.add(op1, op2)

def Pooling(is_train, memcopyin, output_path, 
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, matsize, channels_in, 
        poolsize, strides, padding, 
        activation_fct, elements_matrix, 
        sgd = None, rmsprop = None, 
        adagrad = None, adam = None):
    if is_train:
        return PoolingTrain(memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, matsize, channels_in, 
            poolsize, strides, padding, 
            activation_fct, elements_matrix, 
            sgd, rmsprop, adagrad, adam)
    else:
        return PoolingInf(memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, matsize, channels_in, 
            poolsize, strides, padding, 
            activation_fct, elements_matrix)