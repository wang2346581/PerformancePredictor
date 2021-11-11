import numpy as np
from utils.utils import get_dense_colnames, get_support_activation
from utils.layers.runner import Runner, RunnerRT
from utils.layers.train import TrainCore

__all__ = ['Dense', 'DenseInf', 'DenseTrain', 'DenseRT']

class DenseBase():
    def __init__(self, batchsize, dim_input, dim_output,
        activation_fct, use_bias, **kwargs):
        self._batchsize      = batchsize
        self._dim_input      = dim_input
        self._dim_output     = dim_output
        self._activation_fct = activation_fct
        self._use_bias       = use_bias
        super(DenseBase, self).__init__(**kwargs)
    
    def __str__(self):
        str_ = ""
        for i in get_dense_colnames():
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        return str_
    
    @property
    def batchsize(self):
        return self._batchsize
    
    @property
    def dim_input(self):
        return self._dim_input
    
    @property
    def dim_output(self):
        return self._dim_output
    @property
    def activation_fct(self):
        return self._activation_fct
    
    @property
    def use_bias(self):
        return self._use_bias
    
class DenseInf(DenseBase, Runner):
    def __init__(self, memcopyin, output_path, 
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, dim_input, dim_output,
        activation_fct, use_bias, **kwargs):
        super(DenseInf, self).__init__(memcopyin=memcopyin, output_path=output_path, 
            iter_warmup=iter_warmup, iter_benchmark=iter_benchmark, hashkey=hashkey, 
            batchsize=batchsize, dim_input=dim_input, dim_output=dim_output,
            activation_fct=activation_fct, use_bias=use_bias, **kwargs)

    def gen_input(self):
        import tensorflow as tf
        if not self._input:
            if self.memcopyin:
                self._input = tf.placeholder(tf.float32, shape=[None, self.dim_input], name=self.tensor_input_name)
            else:
                self._input = tf.Variable(tf.random_normal([self.batchsize, self.dim_input]))
    
    def gen_np_input(self):
        self.np_input = np.random.normal(127, 60, (self.batchsize, self.dim_input)).astype(float)
    
    def gen_op(self):
        import tensorflow as tf
        self.gen_input()
        self._op = tf.layers.dense(self.input, 
            units=self.dim_output, kernel_initializer=tf.ones_initializer(),
            activation=eval(get_support_activation()[self.activation_fct]), 
            use_bias=self.use_bias)

    def __str__(self):
        str_ = ""
        for i in get_dense_colnames():
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        str_ += "hashkey: " + str(self.hashkey) + ", "
        str_ += "output_path: " + str(self.output_path)
        return str_
    
class DenseTrain(DenseInf, TrainCore):
    def __init__(self, memcopyin, output_path, 
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, dim_input, dim_output,
        activation_fct, use_bias, 
        sgd, rmsprop, adagrad, adam):

        super(DenseTrain, self).__init__(memcopyin=memcopyin, 
            output_path = output_path, 
            iter_warmup = iter_warmup, iter_benchmark = iter_benchmark, 
            hashkey = hashkey, batchsize=batchsize, 
            dim_input=dim_input, dim_output=dim_output, 
            activation_fct=activation_fct, use_bias=use_bias, 
            sgd=sgd, rmsprop=rmsprop, adagrad=adagrad, adam=adam)

    def gen_output(self):
        import tensorflow as tf
        self.target = tf.Variable(tf.ones([
                    self.batchsize,
                    self.dim_output],
                    dtype=float))
    
    def gen_op(self):
        import tensorflow as tf
        super(DenseTrain, self).gen_op()
        self.gen_output()
        self.gen_opt()
        loss_ = tf.reduce_mean(tf.square(self.op - self.target))
        self.op = self.opt.minimize(loss=loss_)

class DenseRT(DenseBase, RunnerRT):
    def __init__(self, memcopyin, output_path, 
        output_pb_path, output_json_path,
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, dim_input, dim_output,
        activation_fct, use_bias):

        super(DenseRT, self).__init__(memcopyin=memcopyin, 
            output_path = output_path, 
            output_pb_path = output_pb_path, output_json_path = output_json_path,
            iter_warmup=iter_warmup, iter_benchmark=iter_benchmark, hashkey=hashkey, 
            batchsize=batchsize, dim_input=dim_input, dim_output=dim_output,
            activation_fct=activation_fct, use_bias=use_bias)

    def get_batchsize(self):
        return self.batchsize
    
    def gen_input(self):
        import tensorflow as tf
        if not self._input:
            if self.memcopyin:
                self._input = tf.placeholder(tf.float32, shape=[None, self.dim_input], name=self.tensor_input_name)
            else:
                self._input = tf.Variable(tf.random_normal([self.batchsize, self.dim_input]))
    
    def gen_np_input(self):
        self.np_input = np.random.normal(127, 60, (self.batchsize, self.dim_input)).astype(float)
    
    def gen_op(self):
        import tensorflow as tf
        self.gen_input()
        op1 = tf.layers.dense(self.input, 
            units=self.dim_output, kernel_initializer=tf.ones_initializer(),
            activation=eval(get_support_activation()[self.activation_fct]), 
            use_bias=self.use_bias)
        op2 = tf.layers.dense(self.input, 
            units=self.dim_output, kernel_initializer=tf.ones_initializer(),
            activation=eval(get_support_activation()[self.activation_fct]), 
            use_bias=self.use_bias)
        self.op = tf.add(op1, op2)
    
def Dense(is_train, memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, dim_input, dim_output,
            activation_fct, use_bias, 
            sgd = None, rmsprop = None, 
            adagrad = None, adam = None):
    if is_train:
        return DenseTrain(memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, dim_input, dim_output,
            activation_fct, use_bias, 
            sgd, rmsprop, adagrad, adam)
    else:
        return DenseInf(memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, dim_input, dim_output,
            activation_fct, use_bias)