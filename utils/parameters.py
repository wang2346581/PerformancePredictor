import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from .utils import get_colnames, get_hash_colnames, get_support_optimizer, get_support_activation

class NumBase():
    def __init__(self, nums, **kwargs):
        self._nums = nums
        super(NumBase, self).__init__(**kwargs)
     
    @property
    def nums(self):
        return self._nums

    @nums.setter
    def nums(self, nums):
        self._nums = nums

class ParamsBase(NumBase):
    def __init__(self, nums, is_train, type_name, precision = 32, **kwargs):
        self._is_train        = is_train
        self._typename        = type_name
        self._data            = None
        self._batchsize       = None
        self._activation_list = get_support_activation() ### TBD
        self._colnames        = get_colnames(self.typename, self.is_train)
        self._hash_colnames   = get_hash_colnames()
        self._hashkey         = None
        ### TBD ###
        self._precision       = precision
        super(ParamsBase, self).__init__(nums = nums, **kwargs)
 
    def generate_params_with_hashkey(self):
        self.generate_params()
        self.generate_hashkey()

    def generate_hashkey(self):
        if self.data is None:
            print("DataFrame is not found!!")
            return
        
        if self.hash_colnames[0] in list(self.data.columns):
            print("Already has hashkey, Do not genreate it again!")
            return

        self.finetune_column_order()
        print("generate the key, please wait...")
        self._hashkey  = '[' + self.data[self.colnames].astype(str).apply(','.join, axis = 1) + ']' #finetune
        self.data[self.hash_colnames[0]] = self.hashkey

    def set_data(self, df_):
        self._data = df_
        if not self._colnames:
            self.colnames = self.data.columns

    def finetune_column_order(self):
        if self.hash_colnames[0] in list(self.data.columns):
            colnames_with_haskey = self.colnames + [self.hash_colnames[0]]        
            self.data = self._data[colnames_with_haskey]
        else:
            self.data = self._data[self.colnames]
    
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
    
    def generate_params(self):
        raise NotImplementedError("Please Implement this method")
    
    @property
    def data(self):
        return self._data

    @property
    def batchsize(self):
        return self._batchsize

    @property
    def activation_list(self):
        return self._activation_list
    
    @property
    def colnames(self):
        return self._colnames
    
    @property
    def hash_colnames(self):
        return self._hash_colnames

    @property
    def hashkey(self):
        return self._hashkey 

    @property
    def precision(self):
        return self._precision

    @property
    def is_train(self):
        return self._is_train

    @property
    def typename(self):
        return self._typename

    @data.setter
    def data(self, df_):
        if isinstance(df_, pd.DataFrame):
            self._data = df_
            self.colnames = self.data.columns
        else:
            print("Type of data need to set as pd.DataFrame!")

    @colnames.setter
    def colnames(self, colnames):
        self._colnames = colnames
    
class ConvBase(ParamsBase):
    def __init__(self, nums, is_train, **kwargs):
        self._matsize        = None
        self._kernelsize     = None
        self._strides        = None
        self._padding        = None
        self._activation_fct = None
        self._use_bias       = None
        self._elements_matrix = None
        self._elements_kernel = None
        super(ConvBase, self).__init__(nums = nums, is_train = is_train, type_name = 'convolution', **kwargs)

        
    def generate_conv_params(self):
        self._batchsize      = np.random.randint(1,  65, self.nums)
        self._matsize        = np.random.randint(1, 513, self.nums)
        self._kernelsize   = np.zeros(self.nums, dtype=np.int32)
        self._channels_in  = np.zeros(self.nums, dtype=np.int32)
        self._channels_out = np.zeros(self.nums, dtype=np.int32)
        self._strides        = np.random.randint(1,   5, self.nums)
        self._padding        = np.random.randint(0,   2, self.nums)
        self._activation_fct = np.random.randint(0, len(['None', 'tf.nn.relu']), self.nums)
        self._use_bias       = np.random.choice([True, False], self.nums)

        for i in range(self.nums):
            self._kernelsize[i]   = np.random.randint(1, min(7, self.matsize[i])+1)
            self._channels_in[i]  = np.random.randint(1, 10000/self.matsize[i])
            self._channels_out[i] = np.random.randint(1, 10000/self.matsize[i])

        self._elements_matrix = np.square(self.matsize)
        self._elements_kernel = np.square(self.kernelsize)

    def get_input(self, index):
        raise NotImplementedError("Please Implement this method")

    def get_output(self, index):
        raise NotImplementedError("Please Implement this method")
    
    def get_tensor_from_index(self, index):
        raise NotImplementedError("Please Implement this method")

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

class DenseBase(ParamsBase):
    def __init__(self, nums, is_train, **kwargs):
        self._dim_input      = None
        self._dim_output     = None
        self._use_bias       = None
        self._activation_fct = None
        super(DenseBase, self).__init__(nums, is_train, type_name = 'dense', **kwargs)
    
    def generate_dense_params(self):
        self._batchsize      = np.random.randint(1,   65, self.nums)
        self._dim_input      = np.random.randint(1, 32769, self.nums)
        self._dim_output     = np.random.randint(1, 4097, self.nums)
        self._use_bias = np.random.randint(0, 2, self.nums)
        self._activation_fct = np.random.randint(0, len(self.activation_list), self.nums)

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

class PoolingBase(ParamsBase):
    def __init__(self, nums, is_train, **kwargs):
        self._matsize        = None
        self._channels_in    = None
        self._poolsize       = None
        self._strides        = None
        self._padding        = None
        self._activation_fct = None
        self._elements_matrix = None
        super(PoolingBase, self).__init__(nums, is_train, type_name = 'pooling', **kwargs)
    
    def generate_pooling_params(self):
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

class TrainBase(NumBase):
    def __init__(self, nums, **kwargs):
        self._opt_sgd     = None
        self._opt_adagrad = None
        self._opt_rmsprop = None
        self._opt_adam    = None
        super(TrainBase, self).__init__(nums = nums, **kwargs)

    def generate_train_params(self):
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

class ParamsConv(ConvBase):
    def __init__(self, nums):
        super(ParamsConv, self).__init__(nums = nums, is_train = 0)

    def generate_params(self):
        self.generate_conv_params()
        self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.kernelsize, self.channels_in, self.channels_out, 
            self.strides, self.padding, self.activation_fct, self.use_bias, 
            self.elements_matrix, self.elements_kernel,] ).transpose(), axis=0), columns=self.colnames)

class ParamsConvTrain(ConvBase, ParamsBase, TrainBase):
    def __init__(self, nums):
        super(ParamsConvTrain, self).__init__(nums = nums, is_train = 1)
    
    def generate_params(self):
        self.generate_conv_params()
        self.generate_train_params()
        self.data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.kernelsize, self.channels_in, self.channels_out, 
            self.strides, self.padding, self.activation_fct, self.use_bias, 
            self.elements_matrix, self.elements_kernel,
            self.opt_sgd, self.opt_adagrad, self.opt_rmsprop, self.opt_adam]).transpose(), axis=0), columns=self.colnames)
    
class ParamsDense(DenseBase):
    def __init__(self, nums):
        super(ParamsDense, self).__init__(nums, is_train = 0)

    def generate_params(self):
        self.generate_dense_params()
        self.data = pd.DataFrame(np.unique(np.array([self.batchsize, self.dim_input, self.dim_output,
                self.use_bias, self.activation_fct]).transpose(), axis=0), columns=self.colnames)

class ParamsDenseTrain(DenseBase, TrainBase):
    def __init__(self, nums):
        super(ParamsDenseTrain, self).__init__(nums, is_train = 1)

    def generate_params(self):
        self.generate_dense_params()
        self.generate_train_params()
        self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.dim_input, self.dim_output,
            self.use_bias, self.activation_fct,
            self.opt_sgd, self.opt_adagrad, self.opt_rmsprop, self.opt_adam]).transpose(), axis=0), columns=self.colnames)

class ParamsPooling(PoolingBase):
    def __init__(self, nums):
        super(ParamsPooling, self).__init__(nums, is_train = 0)

    def generate_params(self):
        self.generate_pooling_params()
        self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.channels_in,
                self.poolsize, self.strides, self.padding,
                self.elements_matrix]).transpose(), axis=0), columns=self.colnames)    

class ParamsPoolingTrain(PoolingBase, TrainBase):
    def __init__(self, nums):
        super(ParamsPoolingTrain, self).__init__(nums, is_train = 1)

    def generate_params(self):
        self.generate_pooling_params()
        self.generate_train_params()
        self._data = pd.DataFrame(np.unique(np.array([self.batchsize, self.matsize, self.channels_in,
            self.poolsize, self.strides, self.padding, self.elements_matrix, 
            self.opt_sgd, self.opt_adagrad, self.opt_rmsprop, self.opt_adam]).transpose(), axis=0), columns=self.colnames)

def get_params(num, predition_layertype, is_train = 0):
    if predition_layertype == 'convolution':
        return ParamsConvTrain(num) if is_train else ParamsConv(num)
    elif predition_layertype == 'pooling':
        return ParamsPoolingTrain(num) if is_train else ParamsPooling(num)
    elif predition_layertype == 'dense':
        return ParamsDenseTrain(num) if is_train else ParamsDense(num)
    else:
        print("This type of layer is not support!")
        exit()
    return 