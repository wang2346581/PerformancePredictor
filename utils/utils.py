import os
import sys
import json
import shutil
import socket
import logging
from termcolor import colored
from datetime import datetime
import tensorflow as tf

def get_support_accuracy_metrics():
    return ['rmse', 'mape', 'mae', 'r2']

def get_support_devices():
    dict_ = {
        '1080ti': 'gpu',
    }
    return dict_

def get_support_layers():
    return ['convolution', 'pooling', 'dense']

def get_support_models():
    return ['lenet', 'alexnet', 'vgg16', 'vgg19', 'resnet50', 'inceptionv3', 'mobilenet']

def get_verify_models():
    return ['lenet', 'alexnet', 'vgg16']

def get_support_batchsize():
    return [1, 2, 4, 8, 16, 32, 64, 128]

def get_model_csv_total_columns(is_train = 0):
    total_cols = ['layers', 'name', 'operation']
    total_cols = total_cols + [i for i in get_cov_colnames(is_train = is_train)   if i not in total_cols]
    total_cols = total_cols + [i for i in get_dense_colnames(is_train = is_train) if i not in total_cols]
    total_cols = total_cols + [i for i in get_pool_colnames(is_train = is_train)  if i not in total_cols]
    total_cols = total_cols + [i for i in get_time_colnames() if i not in total_cols]
    total_cols = total_cols + [i for i in get_profile_colnames(is_train = is_train)  if i not in total_cols]
    return total_cols

def get_model_predict_total_columns():
    total_cols = ['layers', 'name', 'operation']
    total_cols = total_cols + [i for i in get_cov_colnames()   if i not in total_cols]
    total_cols = total_cols + [i for i in get_dense_colnames() if i not in total_cols]
    total_cols = total_cols + [i for i in get_pool_colnames()  if i not in total_cols]
    total_cols = total_cols + [i for i in get_time_colnames() if i not in total_cols]
    total_cols = total_cols + [i for i in get_profile_colnames()  if i not in total_cols]
    total_cols = total_cols + [i for i in get_predict_time_colnames()  if i not in total_cols]
    return total_cols

def get_end_to_end_model(model):
    if model == 'vgg16':
        from keras.applications.vgg16 import VGG16
        end_to_end_model = VGG16()
    elif model == 'vgg19':
        from keras.applications.vgg19 import VGG19
        end_to_end_model = VGG19()
    elif model == 'resnet50': 
        from keras.applications.resnet50 import ResNet50
        end_to_end_model = ResNet50()
    elif model == 'inceptionv3':
        from keras.applications.inception_v3 import InceptionV3
        end_to_end_model = InceptionV3()
    elif model == 'lenet':
        from keras.models import Sequential
        from keras.layers import Dense,Flatten
        from keras.layers.convolutional import Conv2D,MaxPooling2D
        end_to_end_model = Sequential()
        
        #end_to_end_model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(Conv2D(6,(5,5),strides=(1,1),input_shape=(32,32,1),padding='valid',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(MaxPooling2D(pool_size=(2,2)))
        #end_to_end_model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))

        end_to_end_model.add(Conv2D(16,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(MaxPooling2D(pool_size=(2,2)))
        end_to_end_model.add(Flatten())
        end_to_end_model.add(Dense(120,activation='relu'))
        end_to_end_model.add(Dense(84,activation='relu'))
        end_to_end_model.add(Dense(10,activation='softmax'))
    elif model =='alexnet':
        from keras.models import Sequential
        from keras.layers import Dense,Flatten,Dropout
        from keras.layers.convolutional import Conv2D,MaxPooling2D
        end_to_end_model = Sequential()
        end_to_end_model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(227,227,3),padding='valid',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        end_to_end_model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        end_to_end_model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        end_to_end_model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        end_to_end_model.add(Flatten())
        end_to_end_model.add(Dense(4096,activation='relu'))
        end_to_end_model.add(Dropout(0.5))
        end_to_end_model.add(Dense(4096,activation='relu'))
        end_to_end_model.add(Dropout(0.5))
        end_to_end_model.add(Dense(1000,activation='softmax'))
    elif model == 'mobilenet':
        from keras.applications.mobilenet import MobileNet
        end_to_end_model = MobileNet()
    else:
        print('you should use --model parameter to specify parse which neural network model, ex: --model vgg16')
        exit()
    return end_to_end_model

def get_colnames(typename, is_train = 0):
    if typename == 'convolution':
        return get_cov_colnames(is_train)
    elif typename == 'dense':
        return get_dense_colnames(is_train)
    elif typename == 'pooling':
        return get_pool_colnames(is_train)
    else:
        print("This type of layer is not support!")
        return

def get_hash_colnames():
    return ['hashkey']

def get_support_activation():
    return ['None', 'tf.nn.relu'] 

def get_support_optimizer():
    return ['sgd', 'adagrad', 'rmsprop', 'adam']

def get_cov_colnames(is_train = 0):
    cols_ = ['batchsize', 'matsize', 'kernelsize', 'channels_in', 'channels_out', 'strides', 
        'padding', 'activation_fct', 'use_bias', 'elements_matrix', 'elements_kernel']
    return cols_ + get_support_optimizer() if is_train else cols_

def get_dense_colnames(is_train = 0):
    cols_ = ['batchsize', 'dim_input', 'dim_output', 'use_bias', 'activation_fct']
    return cols_ + get_support_optimizer() if is_train else cols_

def get_pool_colnames(is_train = 0):
    cols_ = ['batchsize', 'matsize', 'channels_in', 'poolsize', 'strides', 'padding', 'elements_matrix']
    return cols_ + get_support_optimizer() if is_train else cols_

def get_time_colnames():
    return ['time_max', 'time_min', 'time_median', 'time_mean', 'time_trim_mean']

def get_profile_colnames(is_train = 0):
    if is_train:
        #return ['preprocess_time', 'execution_time', 'tranposeout_time', 'calculate_time', 'sess_time']
        return ['preprocess_time', 'forward_time', 'backward_time', 'tranposeout_time', 'calculate_time', 'sess_time']
    return ['preprocess_time', 'execution_time', 'memcpy_time', 'retval_time', 'retval_half_time', 
        'memcpy_retval', 'memcpy_retval_half', 'sess_time']#, 'elements_matrix', 'elements_kernel']

def get_predict_time_colnames():
    return ['pre_time_abse', 'pre_time_re', 'pre_time_rmse']


def get_colnames_from_dict(is_train = 0):
    time_colnames    = get_time_colnames()
    profile_colnames = get_profile_colnames(is_train = is_train)
    conv_colnames  = get_cov_colnames(is_train)
    dense_colnames = get_dense_colnames(is_train)
    pool_colnames  = get_pool_colnames(is_train)
    cols_dict = {
        'convolution': conv_colnames,
        'dense': dense_colnames,
        'pooling': pool_colnames,
        'profile': profile_colnames,
        'time': time_colnames,
        'hash': get_hash_colnames()
    }
    return cols_dict

def backup_file(file_path):
    ### Backup the Output CSV file
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    base_name = os.path.basename(file_path)
    path = os.path.dirname(file_path)
    split_basname = os.path.splitext(base_name)
    bk_filename = split_basname[0] + '_' + datetime.now().strftime('%m%d-%H%M%S') + split_basname[1]
    print(warn_tag + 'Ouput CSV: ' + file_path + ' is existed, backup as ' + bk_filename)
    os.rename(file_path, os.path.join(path, bk_filename))

def write_file(data, path, file):
    file_path = os.path.join(path, file)
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    if not os.path.isdir(path):
        os.makedirs(path)
    else:
        if os.path.isfile(file_path):
            backup_file(file_path)

    print(warn_tag + 'Auto create file: ' + file_path)
    data.to_csv(file_path, index=False)

def append_file(data, path, file):
    file_path = os.path.join(path, file)
    data.to_csv(file_path, index=False, mode='a', header=False)

def get_device_limit_memory_size(device):
    device_memory_dict = {
        'P620': 2,
        'P1000': 4,
        'P2000': 5,
        'P4000': 8,
        'P5000': 16,
        '1080ti': 11,
        '2080ti': 11
    }
    return device_memory_dict[device]

def get_feature_target(filename):
    _logger.info("==> get the feature and target...")
    if os.path.isfile(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            feature = data["feature"]
            target = data["target"]
    else:
        _logger.warn("[FT] feature/target json file is not found, use the default opt")
        feature = get_cov_colnames()
        target  = get_time_colnames()[3]
    return feature, target

class _MyFormatter(logging.Formatter):
    """Copy from tensorpack.
    """
    def format(self, record):
        date = colored('IP:%s '%str(ip), 'yellow') + colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
ip = get_ip()


def _getlogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger

_logger = _getlogger()

def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')

def _set_file(path):
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        _logger.info("Existing log file '{}' backuped to '{}'".
            format(path, backup_name))
    hdl = logging.FileHandler(filename=path,
        encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    _logger.addHandler(hdl)
    _logger.info("Argv: " + ' '.join(sys.argv))

def get_support_optimizer_function(opt_):
    #['sgd', 'adagrad', 'rmsprop', 'adam']
    opt_function = None
    if opt_ == 'sgd':
        opt_function = 'GradientDescentOptimizer(learning_rate=0.01)'
    elif opt_ == 'adagrad':
        opt_function = 'AdagradOptimizer(learning_rate=0.01)'
    elif opt_ == 'rmsprop':
        opt_function = 'RMSPropOptimizer(learning_rate=0.01)'
    elif opt_ == 'adam':
        opt_function = 'AdamOptimizer(learning_rate=0.01)'
    return opt_function
