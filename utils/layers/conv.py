import numpy as np
from utils.utils import get_cov_colnames, get_support_activation
from utils.layers.runner import Runner, RunnerRT
from utils.layers.train import TrainCore

__all__ = ['Conv', 'ConvolutionInf', 'ConvolutionTrain', 'ConvolutionRT']

class ConvolutionBase():
    def __init__(self, batchsize, matsize, 
        kernelsize, channels_in, channels_out,
        strides, padding, activation_fct, use_bias, 
        elements_matrix, elements_kernel, **kwargs):
        self._batchsize      = batchsize
        self._matsize        = matsize
        self._kernelsize     = kernelsize
        self._channels_in    = channels_in
        self._channels_out   = channels_out
        self._strides        = strides
        self._padding        = padding
        self._activation_fct = activation_fct
        self._use_bias       = use_bias
        self._elements_matrix = elements_matrix
        self._elements_kernel = elements_kernel
        super(ConvolutionBase, self).__init__(**kwargs)
        
    def __str__(self):
        str_ = ""
        for i in get_cov_colnames():
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        return str_

    @property
    def batchsize(self):
        return self._batchsize

    @property
    def matsize(self):
        return self._matsize
    
    @property
    def kernelsize(self):
        return self._kernelsize
    
    @property
    def channels_in(self):
        return self._channels_in

    @property
    def channels_out(self):
        return self._channels_out

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
    def use_bias(self):
        return self._use_bias
    
    @property
    def elements_matrix(self):
        return self._elements_matrix

    @property
    def elements_kernel(self):
        return self._elements_kernel

class ConvolutionInf(ConvolutionBase, Runner):
    def __init__(self, memcopyin, output_path, 
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, matsize, kernelsize, channels_in, channels_out,
        strides, padding, activation_fct, use_bias, 
        elements_matrix, elements_kernel, **kwargs):
        super(ConvolutionInf, self).__init__(memcopyin=memcopyin, output_path=output_path, 
            iter_warmup=iter_warmup, iter_benchmark=iter_benchmark, hashkey=hashkey, 
            batchsize=batchsize, matsize=matsize,
            kernelsize=kernelsize, channels_in=channels_in, channels_out=channels_out,
            strides=strides, padding=padding, activation_fct=activation_fct, use_bias=use_bias, 
            elements_matrix=elements_matrix, elements_kernel=elements_kernel,
            **kwargs)

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
        self.op = tf.layers.conv2d(self.input, self.channels_out, 
            kernel_size = [self.kernelsize, self.kernelsize],
            strides=(self.strides, self.strides), 
            padding=('SAME' if self.padding == 1 else 'VALID'),
            activation=eval(get_support_activation()[self.activation_fct]), 
            use_bias=self.use_bias)
    
    def __str__(self):
        str_ = ""
        for i in get_cov_colnames():
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        str_ += "hashkey: " + str(self.hashkey) + ", "
        str_ += "output_path: " + str(self.output_path)
        return str_
        
class ConvolutionTrain(ConvolutionInf, TrainCore):
    def __init__(self, memcopyin, output_path, 
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, matsize, kernelsize, channels_in, channels_out,
        strides, padding, activation_fct, use_bias, 
        elements_matrix, elements_kernel,
        sgd, rmsprop, adagrad, adam):

        super(ConvolutionTrain, self).__init__(memcopyin=memcopyin, 
            output_path = output_path, 
            iter_warmup = iter_warmup, iter_benchmark = iter_benchmark, hashkey = hashkey,
            batchsize=batchsize, matsize=matsize,
            kernelsize=kernelsize, channels_in=channels_in, channels_out=channels_out,
            strides=strides, padding=padding, activation_fct=activation_fct, use_bias=use_bias, 
            elements_matrix=elements_matrix, elements_kernel=elements_kernel,
            sgd=sgd, rmsprop=rmsprop, adagrad=adagrad, adam=adam)

    def gen_output(self):
        import tensorflow as tf
        if self.padding == 1:
        #if layer['padding'].astype(int) == 1:
            target_size = np.ceil(np.float(self.matsize/self.strides))
        else:
            target_size = np.ceil(np.float((self.matsize-(self.kernelsize-1))/self.strides))
        self.target = tf.Variable(tf.ones([
                    self.batchsize,
                    int(target_size),
                    int(target_size),
                    self.channels_out],
                    dtype=float))
    
    def gen_op(self):
        import tensorflow as tf
        super(ConvolutionTrain, self).gen_op()
        self.gen_output()
        self.gen_opt()
        loss_ = tf.reduce_mean(tf.square(self.op - self.target))
        self.op = self.opt.minimize(loss=loss_)
    
    def __str__(self):
        str_ = ""
        for i in get_cov_colnames(1):
            str_ += str(i) +  ": " + str(eval('self.'+str(i))) + ", "
        str_ += "hashkey: " + str(self.hashkey) + ", "
        str_ += "output_path: " + str(self.output_path)
        return str_

class ConvolutionRT(ConvolutionBase, RunnerRT):
    def __init__(self, memcopyin, output_path, 
        output_pb_path, output_json_path,
        iter_warmup, iter_benchmark, hashkey, 
        batchsize, matsize, kernelsize, channels_in, channels_out,
        strides, padding, activation_fct, use_bias, 
        elements_matrix, elements_kernel):

        super(ConvolutionRT, self).__init__(memcopyin=memcopyin, 
            output_path = output_path, 
            output_pb_path = output_pb_path, output_json_path = output_json_path,
            iter_warmup = iter_warmup, iter_benchmark = iter_benchmark, hashkey = hashkey,
            batchsize=batchsize, matsize=matsize,
            kernelsize=kernelsize, channels_in=channels_in, channels_out=channels_out,
            strides=strides, padding=padding, activation_fct=activation_fct, use_bias=use_bias, 
            elements_matrix=elements_matrix, elements_kernel=elements_kernel)
    
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
        op1 = tf.layers.conv2d(self.input, self.channels_out, 
            kernel_size = [self.kernelsize, self.kernelsize],
            strides=(self.strides, self.strides), 
            padding=('SAME' if self.padding == 1 else 'VALID'),
            activation=eval(get_support_activation()[self.activation_fct]), 
            use_bias=self.use_bias)
        op2 = tf.layers.conv2d(self.input, self.channels_out, 
            kernel_size = [self.kernelsize, self.kernelsize],
            strides=(self.strides, self.strides), 
            padding=('SAME' if self.padding == 1 else 'VALID'),
            activation=eval(get_support_activation()[self.activation_fct]), 
            use_bias=self.use_bias)
        op1 = tf.contrib.layers.flatten(op1)
        op2 = tf.contrib.layers.flatten(op2) 
        self.op  = tf.add(op1, op2)
        
def Conv(is_train, memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, matsize, kernelsize,
            channels_in, channels_out, strides, 
            padding, activation_fct, use_bias, 
            elements_matrix, elements_kernel, 
            sgd = None, rmsprop = None, 
            adagrad = None, adam = None):
    if is_train:
        return ConvolutionTrain(memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, matsize, kernelsize,
            channels_in, channels_out, strides, 
            padding, activation_fct, use_bias, 
            elements_matrix, elements_kernel, 
            sgd, rmsprop, adagrad, adam)
    else:
        return ConvolutionInf(memcopyin, output_path, 
            iter_warmup, iter_benchmark, hashkey, 
            batchsize, matsize, kernelsize,
            channels_in, channels_out, strides, 
            padding, activation_fct, use_bias, 
            elements_matrix, elements_kernel)
