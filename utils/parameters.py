import os
import numpy as np
import pandas as pd
import tensorflow as tf
from abc import abstractmethod, ABCMeta
from sklearn.utils import shuffle
from .utils import get_colnames, get_hash_colnames, get_support_optimizer, get_support_optimizer_function

def get_params(num, predition_layertype, is_train = 0):
    if predition_layertype == 'convolution':
        params = ParamsConv(num, predition_layertype, is_train = is_train)
    elif predition_layertype == 'pooling':
        params = ParamsPooling(num, predition_layertype, is_train = is_train)
    elif predition_layertype == 'dense':
        params = ParamsDense(num, predition_layertype, is_train = is_train)
    else:
        print("This type of layer is not support!")
        exit()
    return params

class ParamsBase(metaclass=ABCMeta):
    """Basic paramter of various layer tpye"""
    def __init__(self, nums, typename = '', precision = 32, optimizer = None, memcopyin = 1, is_train = 0):
        self._nums            = nums
        self._typename        = typename
        self._data            = None
        self._batchsize       = None
        self._activation_list = ['None', 'tf.nn.relu']
        self._colnames        = get_colnames(self.typename, is_train)
        self._hash_colnames   = get_hash_colnames()
        self._hashkey         = None

        ### New Feature ###
        self._is_train      = is_train
        self._opt_sgd       = None
        self._opt_adagrad   = None
        self._opt_rmsprop   = None
        self._opt_adam      = None

        ### TBD ###
        self._precision       = precision
        self._optimizer       = optimizer
        self._memcopyin       = memcopyin                 

    def generate_params_with_hashkey(self):
        self.generate_params()
        self.generate_hashkey()

    def generate_hashkey(self):
        if self._data is None:
            print("DataFrame is not found!!")
            return
        
        if self.hash_colnames[0] in list(self.data.columns):
            print("Already has hashkey, Do not genreate it again!")
            self.finetune_column_order()
            return

        print("generate the key, please wait...")
        #self._hashkey  = self.data.apply(get_all_data, colnames=self.colnames, axis = 1) ### old version, but too slow
        self._hashkey  = '[' + self.data[self.colnames].astype(str).apply(','.join, axis = 1) + ']' #finetune
        self._data[self.hash_colnames[0]] = self.hashkey
    
    def set_data(self, df_):
        self._data = df_
        if not self._nums:
            self._nums = self._data.shape[0]
    
    def set_colnames(self, colnames):
        self._colnames = colnames
    
    def set_memcopyin(self, memcopyin):
        self._memcopyin = memcopyin
    
    def finetune_column_order(self):
        if self.hash_colnames[0] in list(self.data.columns):
            colnames_with_haskey = self.colnames + [self.hash_colnames[0]]        
            self._data = self._data[colnames_with_haskey]
        else:
            self._data = self._data[self.colnames]
        
    def auto_generate_elements(self):
        matsize = 'matsize'
        kersize = 'kernelsize'
        ele_mat = 'elements_matrix'
        ele_ker = 'elements_kernel'
        if matsize in self.data.columns:
            if ele_mat not in self.data.columns:
                self._data[ele_mat] =  np.square(self._data[matsize])
        if kersize in self.data.columns:
            if ele_ker not in self.data.columns:
                self._data[ele_ker] =  np.square(self._data[kersize])
        
    def get_shuffle_data(self):
        if self._data is not None:
            return shuffle(self.data).reset_index(drop=True)
        return None

    @abstractmethod
    def generate_params(self):
        '''please Implement it in subclass'''
    
    @abstractmethod
    def get_input(self, index):
        '''please Implement it in subclass'''

    @abstractmethod
    def get_tensor_from_index(self, index):
        '''please Implement it in subclass'''

    @property
    def nums(self):
        return self._nums

    @property
    def typename(self):
        return self._typename
    
    @property
    def data(self):
        return self._data

    @property
    def batchsize(self):
        return self._batchsize.astype(int)

    @property
    def activation_list(self):
        return self._activation_list
    
    @property
    def colnames(self):
        return self._colnames
    
    @property
    def hash_colnames(self):
        if not self._hash_colnames:
            self._hash_colnames = get_hash_colnames()
        else:
            return self._hash_colnames

    @property
    def hashkey(self):
        return self._hashkey 

    @property
    def precision(self):
        return self._precision
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def memcopyin(self):
        return self._memcopyin
    
    @property
    def is_train(self):
        return self._is_train
    
    @property
    def opt_sgd(self):
        return self._opt_sgd.astype(int)
    
    @property
    def opt_adagrad(self):
        return self._opt_adagrad.astype(int)

    @property
    def opt_rmsprop(self):
        return self._opt_rmsprop.astype(int)
    
    @property
    def opt_adam(self):
        return self._opt_adam.astype(int)

class ParamsConv(ParamsBase):
    def __init__(self, nums, typename = 'convolution', precision = 32, optimizer = None, memcopyin = 1, is_train = 0):
        self._matsize        = None
        self._kernelsize     = None
        self._channels_in    = None
        self._channels_out   = None
        self._strides        = None
        self._padding        = None
        self._activation_fct = None
        self._use_bias       = None
        self._elements_matrix = None
        self._elements_kernel = None
        super().__init__(nums, typename, precision, optimizer, memcopyin, is_train)

    def generate_params(self):
        self._batchsize      = np.random.randint(1,  65, self.nums)
        self._matsize        = np.random.randint(1, 513, self.nums)
        self._kernelsize   = np.zeros(self.nums, dtype=np.int32)
        self._channels_in  = np.zeros(self.nums, dtype=np.int32)
        self._channels_out = np.zeros(self.nums, dtype=np.int32)
        self._strides        = np.random.randint(1,   5, self.nums)
        self._padding        = np.random.randint(0,   2, self.nums)
        self._activation_fct = np.random.randint(0, len(self.activation_list), self.nums)
        self._use_bias       = np.random.choice([True, False], self.nums)

        for i in range(self.nums):
            self._kernelsize[i]   = np.random.randint(1, min(7, self.matsize[i])+1)
            self._channels_in[i]  = np.random.randint(1, 10000/self.matsize[i])
            self._channels_out[i] = np.random.randint(1, 10000/self.matsize[i])

        self._elements_matrix = np.square(self.matsize)
        self._elements_kernel = np.square(self.kernelsize)

        ### New Features ###
        if self.is_train:
            opt_size = len(get_support_optimizer())
            opt_rand = np.random.randint(0, opt_size, self.nums)
            self._opt_sgd     = np.zeros(self.nums)
            self._opt_adagrad = np.zeros(self.nums)
            self._opt_rmsprop = np.zeros(self.nums)
            self._opt_adam    = np.zeros(self.nums)
            self._opt_sgd[np.where( opt_rand == 0 )]     = 1 ## sgd is 1, ada is 2, ..., etc
            self._opt_adagrad[np.where( opt_rand == 1 )] = 1
            self._opt_rmsprop[np.where( opt_rand == 2 )] = 1
            self._opt_adam[np.where( opt_rand == 3 )]    = 1
            self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.kernelsize, self.channels_in, self.channels_out, 
                self.strides, self.padding, self.activation_fct, self.use_bias, 
                self.elements_matrix, self.elements_kernel, 
                self.opt_sgd, self.opt_adagrad, self.opt_rmsprop, self.opt_adam]).transpose(), axis=0), columns=self.colnames)
        else:
            self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.kernelsize, self.channels_in, self.channels_out, 
                self.strides, self.padding, self.activation_fct, self.use_bias, 
                self.elements_matrix, self.elements_kernel]).transpose(), axis=0), columns=self.colnames)

    def get_input(self, index):
        layer = self.data.loc[index, :]
        input_ = np.random.normal(127, 60, (layer['batchsize'], layer['matsize'], layer['matsize'], layer['channels_in'])).astype(float)
        return input_
    
    def get_output(self, index):
        layer = self.data.loc[index, :]
        if layer['padding'].astype(int) == 1:
            target_size = np.ceil(np.float(layer['matsize'].astype(int)/layer['strides'].astype(int)))
        else:
            target_size = np.ceil(np.float((layer['matsize'].astype(int)-(layer['kernelsize'].astype(int)-1)))/layer['strides'].astype(int))
        target = tf.Variable(tf.ones([
                    layer['batchsize'].astype(int),
                    int(target_size),
                    int(target_size),
                    layer['channels_out'].astype(int)],
                    dtype=float))
        return target
    
    def get_tensor_from_index(self, index):
        layer = self.data.loc[index, :]
        opt  = None
        if self.is_train:
            target = self.get_output(index)
            list_opts = get_support_optimizer()
            opt_function_name = None
            for opt in list_opts:
                if layer[opt].astype(int) == 1:
                    opt_function_name = get_support_optimizer_function(opt)
                    break
            opt = eval('tf.train.{}'.format(opt_function_name))
            #print("opt func name:{}, tf func: {}".format(opt_function_name, opt))
        
        if self.memcopyin:
            self.inputs = tf.placeholder(tf.float32, shape=[None, layer['matsize'].astype(int), 
                layer['matsize'].astype(int), layer['channels_in'].astype(int)], name="inputs")
        else:
            self.inputs = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
                layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
      
        op = tf.layers.conv2d(self.inputs, filters=layer['channels_out'].astype(int), 
            kernel_size=[layer['kernelsize'].astype(int), layer['kernelsize'].astype(int)], 
            strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
            padding=('SAME' if layer['padding'].astype(int) ==1 else 'VALID'),
            activation=eval(self.activation_list[layer['activation_fct'].astype(int)]), 
            use_bias=layer['use_bias'].astype(int),
            name=self.typename)
        
        if self.is_train:
            loss_ = tf.reduce_mean( tf.square( op - target ) )
            train_op = opt.minimize(loss=loss_)
            return train_op
        return op

    @property
    def matsize(self):
        return self._matsize.astype(int)
    
    @property
    def kernelsize(self):
        return self._kernelsize.astype(int)
    
    @property
    def channels_in(self):
        return self._channels_in.astype(int)

    @property
    def channels_out(self):
        return self._channels_out.astype(int)

    @property
    def strides(self):
        return self._strides.astype(int)
    
    @property
    def padding(self):
        return self._padding

    @property
    def activation_fct(self):
        return self._activation_fct.astype(int)

    @property
    def use_bias(self):
        return self._use_bias.astype(int)
    
    @property
    def elements_matrix(self):
        return self._elements_matrix.astype(int)

    @property
    def elements_kernel(self):
        return self._elements_kernel.astype(int)

class ParamsDense(ParamsBase):
    def __init__(self, nums, typename = 'dense', precision = 32, optimizer = None, memcopyin = 1, is_train = 0):
        self._dim_input      = None
        self._dim_output     = None
        self._use_bias       = None
        self._activation_fct = None
        super().__init__(nums, typename, precision, optimizer, memcopyin, is_train)
    
    def generate_params(self):
        self._batchsize      = np.random.randint(1,   65, self.nums)
        self._dim_input      = np.random.randint(1, 32769, self.nums)
        self._dim_output     = np.random.randint(1, 4097, self.nums)
        self._use_bias = np.random.randint(0, 2, self.nums)
        self._activation_fct = np.random.randint(0, len(self.activation_list), self.nums)

        ### New Features ###
        if self.is_train:
            opt_size = len(get_support_optimizer())
            opt_rand = np.random.randint(0, opt_size, self.nums)
            self._opt_sgd     = np.zeros(self.nums)
            self._opt_adagrad = np.zeros(self.nums)
            self._opt_rmsprop = np.zeros(self.nums)
            self._opt_adam    = np.zeros(self.nums)
            self._opt_sgd[np.where( opt_rand == 0 )]     = 1 ## sgd is 1, ada is 2, ..., etc
            self._opt_adagrad[np.where( opt_rand == 1 )] = 1
            self._opt_rmsprop[np.where( opt_rand == 2 )] = 1
            self._opt_adam[np.where( opt_rand == 3 )]    = 1
            self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.dim_input, self.dim_output,
                self.use_bias, self.activation_fct,
                self.opt_sgd, self.opt_adagrad, self.opt_rmsprop, self.opt_adam]).transpose(), axis=0), columns=self.colnames)
        else:
            self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.dim_input, self.dim_output,
                self.use_bias, self.activation_fct]).transpose(), axis=0), columns=self.colnames)
    
    def get_input(self, index):
        layer = self.data.loc[index, :]
        input_ = np.random.normal(127, 60, (layer['batchsize'], layer['dim_input'])).astype(float)
        return input_

     
    def get_output(self, index):
        layer = self.data.loc[index, :]
    
        target = tf.Variable(tf.ones([
                    layer['batchsize'].astype(int),
                    layer['dim_output'].astype(int)],
                    dtype=float))
        return target

    def get_tensor_from_index(self, index):
        layer = self.data.loc[index, :]
        
        if self.is_train:
            target = self.get_output(index)
            list_opts = get_support_optimizer()
            opt_function_name = None
            for opt in list_opts:
                if layer[opt].astype(int) == 1:
                    opt_function_name = get_support_optimizer_function(opt)
                    break
            opt = eval('tf.train.{}'.format(opt_function_name))
            #print("opt func name:{}, tf func: {}".format(opt_function_name, opt))

        if self.memcopyin:
            self.inputs= tf.placeholder(tf.float32, shape=[None, layer['dim_input'].astype(int)], name="inputs")
        else:
            self.inputs = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), layer['dim_input'].astype(int)]))
        op = tf.layers.dense(self.inputs, units=layer['dim_output'].astype(int),
            kernel_initializer=tf.ones_initializer(), use_bias=layer['use_bias'].astype(int),
            activation=eval(self.activation_list[layer['activation_fct'].astype(int)]), 
            name = self.typename)
        
        if self.is_train:
            loss_ = tf.reduce_mean( tf.square( op - target ) )
            train_op = opt.minimize(loss=loss_)
            return train_op
        return op

    @property
    def dim_input(self):
        return self._dim_input.astype(int)
    
    @property
    def dim_output(self):
        return self._dim_output.astype(int)
    @property
    def activation_fct(self):
        return self._activation_fct.astype(int)
    
    @property
    def use_bias(self):
        return self._use_bias.astype(int)

class ParamsPooling(ParamsBase):
    def __init__(self, nums, typename = 'pooling', precision = 32, optimizer = None, memcopyin = 1, is_train = 0):
        self._matsize        = None
        self._channels_in    = None
        self._poolsize       = None
        self._strides        = None
        self._padding        = None
        self._activation_fct = None
        self._elements_matrix = None
        super().__init__(nums, typename, precision, optimizer, memcopyin, is_train)

    def generate_params(self):
        self._batchsize = np.random.randint(1,  65, self.nums)
        self._matsize   = np.random.randint(1, 513, self.nums)
        self._channels_in = np.zeros(self.nums, dtype=np.int32)
        self._poolsize    = np.zeros(self.nums, dtype=np.int32)
        self._strides   = np.random.randint(1,   5, self.nums)
        self._padding   = np.random.randint(0,   2, self.nums)
        
        self._activation_fct = np.random.randint(0,len(self.activation_list), self.nums)
        
        for i in range(self.nums):
            self._channels_in[i] = np.random.randint(1, 10000/self.matsize[i])
            self._poolsize[i]    = np.random.randint(1, min(7, self.matsize[i])+1)
        
        self._elements_matrix = np.square(self.matsize)
        
        ### New Features ###
        if self.is_train:
            opt_size = len(get_support_optimizer())
            opt_rand = np.random.randint(0, opt_size, self.nums)
            self._opt_sgd     = np.zeros(self.nums)
            self._opt_adagrad = np.zeros(self.nums)
            self._opt_rmsprop = np.zeros(self.nums)
            self._opt_adam    = np.zeros(self.nums)
            self._opt_sgd[np.where( opt_rand == 0 )]     = 1 ## sgd is 1, ada is 2, ..., etc
            self._opt_adagrad[np.where( opt_rand == 1 )] = 1
            self._opt_rmsprop[np.where( opt_rand == 2 )] = 1
            self._opt_adam[np.where( opt_rand == 3 )]    = 1
            self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.channels_in,
                self.poolsize, self.strides, self.padding, self.elements_matrix, 
                self.opt_sgd, self.opt_adagrad, self.opt_rmsprop, self.opt_adam]).transpose(), axis=0), columns=self.colnames)
        else:
            self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.channels_in,
                self.poolsize, self.strides, self.padding,
                self.elements_matrix]).transpose(), axis=0), columns=self.colnames)    

    def get_input(self, index):
        layer = self.data.loc[index, :]
        input_ = np.random.normal(127, 60, (layer['batchsize'], layer['matsize'], layer['matsize'], layer['channels_in'])).astype(float)
        return input_
   
    def get_output(self, index):
        layer = self.data.loc[index, :]
        if layer['padding'].astype(int) == 1:
            target_size = np.ceil(np.float(layer['matsize'].astype(int)/layer['strides'].astype(int)))
        else:
            target_size = np.ceil(np.float((layer['matsize'].astype(int)-(layer['poolsize'].astype(int)-1)))/layer['strides'].astype(int))
        target = tf.Variable(tf.ones([
                    layer['batchsize'].astype(int),
                    int(target_size),
                    int(target_size),
                    layer['channels_in'].astype(int)],
                    dtype=float))
        return target

    def get_tensor_from_index(self, index):
        layer = self.data.loc[index, :]
        opt  = None
        if self.is_train:
            target = self.get_output(index)

            list_opts = get_support_optimizer()
            opt_function_name = None
            #print(self.colnames, list_opts)
            for opt in list_opts:
                #print("-------------------> layer[opt].astype(int):", layer[opt].astype(int) )
                if layer[opt].astype(int) == 1:
                    opt_function_name = get_support_optimizer_function(opt)
                    break
            opt = eval('tf.train.{}'.format(opt_function_name))
            #print("opt func name:{}, tf func: {}".format(opt_function_name, opt))
        
        if self.memcopyin:
            self.inputs = tf.placeholder(tf.float32, shape=[None, layer['matsize'].astype(int), 
                layer['matsize'].astype(int), layer['channels_in'].astype(int)], name="inputs")
        else:
            self.inputs = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
                layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
        op = tf.layers.max_pooling2d(self.inputs, pool_size=(layer['poolsize'].astype(int), layer['poolsize'].astype(int)), 
            strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
            padding=('SAME' if layer['padding'].astype(int)==1 else 'VALID'), 
            name = self.typename)
                
        if self.is_train:
            loss_ = tf.reduce_mean( tf.square( op - target ) )
            train_op = opt.minimize(loss=loss_)
            return train_op
        
        return op

    @property
    def matsize(self):
        return self._matsize.astype(int)
    
    @property
    def channels_in(self):
        return self._channels_in.astype(int)
    
    @property
    def poolsize(self):
        return self._poolsize.astype(int)
    
    @property
    def strides(self):
        return self._strides.astype(int)

    @property
    def padding(self):
        return self._padding

    @property
    def elements_matrix(self):
        return self._elements_matrix.astype(int)

