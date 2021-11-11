from .conv import *
from .dense import *
from .pool import *

import os
from pkgutil import iter_modules

__all__ = ['get_layer', 'get_trt_layer']

def _global_import(name, all_list):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else []
    if lst:
        del globals()[name]
        for k in lst:
            globals()[k] = p.__dict__[k]
            all_list.append(k)

_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules([_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    _global_import(module_name, __all__)

def get_layer(flags, row_data, output_path):
    layer = None
    if flags.predition_layertype == "convolution":
        layer = Conv(flags.is_train, flags.memory_copy_in,
            output_path    = output_path,
            iter_warmup    = flags.iter_warmup,
            iter_benchmark = flags.iter_benchmark,
            hashkey        = row_data['hashkey'],
            batchsize      = row_data['batchsize'],
            matsize        = row_data['matsize'],
            kernelsize     = row_data['kernelsize'],
            channels_in    = row_data['channels_in'],
            channels_out   = row_data['channels_out'],
            strides        = row_data['strides'],
            padding        = row_data['padding'],
            activation_fct = row_data['activation_fct'],
            use_bias       = row_data['use_bias'],
            elements_matrix = row_data['elements_matrix'],
            elements_kernel = row_data['elements_kernel'],
            sgd     = row_data['sgd']     if 'sgd' in row_data.index else None,
            rmsprop = row_data['rmsprop'] if 'rmsprop' in row_data.index else None,
            adagrad = row_data['adagrad'] if 'adagrad' in row_data.index else None,
            adam    = row_data['adam']    if 'adam' in row_data.index else None)

    elif flags.predition_layertype == "dense":
        layer = Dense(flags.is_train, flags.memory_copy_in,
            output_path    = output_path,
            iter_warmup    = flags.iter_warmup,
            iter_benchmark = flags.iter_benchmark,
            hashkey        = row_data['hashkey'],
            batchsize      = row_data['batchsize'],
            dim_input      = row_data['dim_input'],
            dim_output     = row_data['dim_output'],
            activation_fct = row_data['activation_fct'],
            use_bias       = row_data['use_bias'],
            sgd     = row_data['sgd']     if 'sgd' in row_data.index else None,
            rmsprop = row_data['rmsprop'] if 'rmsprop' in row_data.index else None,
            adagrad = row_data['adagrad'] if 'adagrad' in row_data.index else None,
            adam    = row_data['adam']    if 'adam' in row_data.index else None)

    elif flags.predition_layertype == "pooling":
        layer = Pooling(flags.is_train, flags.memory_copy_in, 
            output_path    = output_path,
            iter_warmup    = flags.iter_warmup,
            iter_benchmark = flags.iter_benchmark,
            hashkey        = row_data['hashkey'],
            batchsize      = row_data['batchsize'],
            matsize        = row_data['matsize'],
            channels_in    = row_data['channels_in'],
            poolsize       = row_data['poolsize'],
            strides        = row_data['strides'],
            padding        = row_data['padding'],
            activation_fct = row_data['activation_fct'] if 'activation_fct' in row_data.index else None,
            elements_matrix = row_data['elements_matrix'],
            sgd     = row_data['sgd']     if 'sgd' in row_data.index else None,
            rmsprop = row_data['rmsprop'] if 'rmsprop' in row_data.index else None,
            adagrad = row_data['adagrad'] if 'adagrad' in row_data.index else None,
            adam    = row_data['adam']    if 'adam' in row_data.index else None)
    else:
        print("This type of layer is not support!")
        exit()
    return layer

def get_trt_layer(flags, row_data, output_path):
    layer = None
    if flags.predition_layertype == "convolution":
        layer = ConvolutionRT(
            flags.memory_copy_in,
            output_path     = output_path,
            output_pb_path  = flags.trt_frozen_pb_path,
            output_json_path = flags.trt_frozen_json_path,
            iter_warmup    = flags.iter_warmup,
            iter_benchmark = flags.iter_benchmark,
            hashkey        = row_data['hashkey'],
            batchsize      = row_data['batchsize'],
            matsize        = row_data['matsize'],
            kernelsize     = row_data['kernelsize'],
            channels_in    = row_data['channels_in'],
            channels_out   = row_data['channels_out'],
            strides        = row_data['strides'],
            padding        = row_data['padding'],
            activation_fct = row_data['activation_fct'],
            use_bias       = row_data['use_bias'],
            elements_matrix = row_data['elements_matrix'],
            elements_kernel = row_data['elements_kernel'])

    elif flags.predition_layertype == "dense":
        layer = DenseRT(
            flags.memory_copy_in,
            output_path     = output_path,
            output_pb_path  = flags.trt_frozen_pb_path,
            output_json_path = flags.trt_frozen_json_path,
            iter_warmup    = flags.iter_warmup,
            iter_benchmark = flags.iter_benchmark,
            hashkey        = row_data['hashkey'],
            batchsize      = row_data['batchsize'],
            dim_input      = row_data['dim_input'],
            dim_output     = row_data['dim_output'],
            activation_fct = row_data['activation_fct'],
            use_bias       = row_data['use_bias'])

    elif flags.predition_layertype == "pooling":
        layer = PoolingRT(
            flags.memory_copy_in,
            output_path     = output_path,
            output_pb_path  = flags.trt_frozen_pb_path,
            output_json_path = flags.trt_frozen_json_path,
            iter_warmup    = flags.iter_warmup,
            iter_benchmark = flags.iter_benchmark,
            hashkey        = row_data['hashkey'],
            batchsize      = row_data['batchsize'],
            matsize        = row_data['matsize'],
            channels_in    = row_data['channels_in'],
            poolsize       = row_data['poolsize'],
            strides        = row_data['strides'],
            padding        = row_data['padding'],
            activation_fct = row_data['activation_fct'] if 'activation_fct' in row_data.index else None,
            elements_matrix = row_data['elements_matrix'])
    else:
        print("This type of layer is not support!")
        exit()
    return layer

print(__all__)