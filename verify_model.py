import os
import re
import sys
import time
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from termcolor import colored
from tensorflow.python.client import timeline

from utils.utils import *
from utils.model import Model
from utils.network import get_nn_list
from utils.meter import RegErrorMeter, ExpRegErrorMeter, DExpRegErrorMeter
from utils.timeline_struct import Recorders

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from train_model import get_meter

activation_list = ['None', 'tf.nn.relu']

def pred_data_preparation(flags, df_test, feature=list(), target=str()):
    print("==> Do data preparation ...")
    ### Get Data from csv
    df_train = pd.read_csv(flags.train_csv)
   
    if target not in df_train.columns:
        return None, None, None, None

    # df_test  = pd.read_csv(flags.test_csv)
    if not feature or not target: ### For Fool Proofing
        feature, target = get_feature_target(os.path.join(os.getcwd(), "Feature_Target", "ftf1.json"))
    ### Feature transformations
    ft_str = "[Features] Transformation: {} to Features: {}".format(flags.feature_transform, flags.ft_list)
    if flags.feature_transform == "":
        ft_str = "[Features] No transformation to the Features"
    elif flags.feature_transform == "log":
        for ft in flags.ft_list:
            df_train[ft] = np.log(df_train[ft])
            df_test[ft]  = np.log(df_test[ft])
    elif flags.feature_transform == "boxcox":
        for ft in flags.ft_list:
            df_train[ft], maxlog = stats.boxcox(df_train[ft])
            df_test[ft]  =  stats.boxcox(df_test[ft], maxlog)
    elif flags.feature_transform == "sqrt":
        for ft in flags.ft_list:
            df_train[ft] = np.sqrt(df_train[ft])
            df_test[ft]  = np.sqrt(df_test[ft])
    print(ft_str)
    
    ### Feature Polynormial
    fp_str = "[Features] Polynormial: {} to all Features".format(flags.poly)
    if flags.poly < 2:
        fp_str = "[Features] No polynormial to all Features"
        train_f = df_train[feature]
        test_f  = df_test[feature]
    else:
        poly = PolynomialFeatures(flags.poly)
        train_f = poly.fit_transform(df_train[feature])
        test_f  = poly.fit_transform(df_test[feature])
    print(fp_str)
    ### Feature StandardScale
    print("[Features] Standardscale to all Features")
    scaler = StandardScaler()
    scaler.fit(train_f.astype(float))
    train_scale = scaler.transform(train_f[feature].astype(float))
    test_scale  = scaler.transform(test_f[feature].astype(float))
     ### Target Transformation
    print("[Target] target time * {}".format(flags.magic_scaler)) 
    df_train[target] = df_train[target] * flags.magic_scaler ###for convenience
    df_test[target]  = df_test[target]  * flags.magic_scaler ###for convenience

    if flags.target_transform == 'log':
        print("[Target] logarithm the traget time ") 
        df_train[target] = np.log(df_train[target])
        df_test[target]  = np.log(df_test[target])
    return train_scale, df_train[target], test_scale, df_test[target]#, df_train

def create_conv2d(op, layer, layer_type, dict_layernum, is_train = 0):
    layer_name = layer_type + str(dict_layernum[layer_type] + 1)
    if is_train:
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
        list_opts = get_support_optimizer()
        opt_function_name = None
        for opt in list_opts:
            if layer[opt].astype(int) == 1:
                opt_function_name = get_support_optimizer_function(opt)
                break
        opt = eval('tf.train.{}'.format(opt_function_name))

    if op == None:
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
            layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
   
    op = tf.layers.conv2d(op, filters=layer['channels_out'].astype(int), 
        kernel_size=[layer['kernelsize'].astype(int), layer['kernelsize'].astype(int)], 
        strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
        padding=('SAME' if layer['padding'].astype(int) ==1 else 'VALID'),
        activation=eval(activation_list[layer['activation_fct'].astype(int)]), 
        use_bias=layer['use_bias'].astype(int),
        name=layer_name)
    
    dict_layernum[layer_type] += 1
    
    if is_train:
        loss_ = tf.reduce_mean( tf.square( op - target ) )
        train_op = opt.minimize(loss=loss_)
        return train_op

    return op

def create_pooling(op, layer, layer_type, dict_layernum, is_train = 0):
    layer_name = layer_type + str(dict_layernum[layer_type] + 1)
    if is_train:
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
        list_opts = get_support_optimizer()
        opt_function_name = None
        for opt in list_opts:
            if layer[opt].astype(int) == 1:
                opt_function_name = get_support_optimizer_function(opt)
                break
        opt = eval('tf.train.{}'.format(opt_function_name))

    if op == None:
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), 
            layer['matsize'].astype(int), layer['matsize'].astype(int), layer['channels_in'].astype(int)]))
    
    op = tf.layers.max_pooling2d(op, pool_size=(layer['poolsize'].astype(int), layer['poolsize'].astype(int)), 
        strides=(layer['strides'].astype(int), layer['strides'].astype(int)), 
        padding=('SAME' if layer['padding'].astype(int)==1 else 'VALID'), 
        name = layer_name)

    dict_layernum[layer_type] += 1

    if is_train:
        loss_ = tf.reduce_mean( tf.square( op - target ) )
        train_op = opt.minimize(loss=loss_)
        return train_op

    return op

def create_dense(op, layer, layer_type, dict_layernum, is_train = 0):
    layer_name = layer_type + str(dict_layernum[layer_type] + 1)
    if is_train:
        target = tf.Variable(tf.ones([
                    layer['batchsize'].astype(int),
                    layer['dim_output'].astype(int)],
                    dtype=float))
        list_opts = get_support_optimizer()
        opt_function_name = None
        for opt in list_opts:
            if layer[opt].astype(int) == 1:
                opt_function_name = get_support_optimizer_function(opt)
                break
        opt = eval('tf.train.{}'.format(opt_function_name))
    if op == None:
        op = tf.Variable(tf.random_normal([layer['batchsize'].astype(int), layer['dim_input'].astype(int)]))

    op = tf.layers.dense(inputs=op, units=layer['dim_output'].astype(int),
        kernel_initializer=tf.ones_initializer(), 
        activation=eval(activation_list[layer['activation_fct'].astype(int)]), 
        name = layer_name)

    dict_layernum[layer_type] += 1

    if is_train:
        loss_ = tf.reduce_mean( tf.square( op - target ) )
        train_op = opt.minimize(loss=loss_)
        return train_op

    return op

def pred_validation(flags, model, ckpt_path_name, testdata, testlabel): 
    print("==> Do inference...")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    te_num_datapoints = testdata.shape[0]
    te_list_datapoints = np.arange(0,te_num_datapoints)
    te_num_batches = np.int(np.ceil(te_num_datapoints/model.batch_size))
    with tf.Session() as sess:
        sess.run(init)
        print("==> Resuming model from checkpoint..")
        print(ckpt_path_name)
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path_name))
        regm_test = get_meter(flags.loss_name, flags.target_transform, magic_scaler = flags.magic_scaler)

        for i in range(0, te_num_batches):
            cur_batch_size = min((i+1)*model.batch_size,te_num_datapoints) - (i*model.batch_size)
            batch = te_list_datapoints[i*model.batch_size:min((i+1)*model.batch_size,te_num_datapoints)]
            testloss, test_pred_time = sess.run(
                [model.loss, model.prediction],
                    feed_dict={
                        model.tf_inputs: testdata[batch,:],
                        model.tf_targets: testlabel[batch],
                        model.tf_istraining: False})
            # update to meter
            regm_test.error.update(testlabel[batch], test_pred_time)
            regm_test.loss.update(testloss, cur_batch_size)
        regm_test.error.summary()
    return regm_test.error.prediction, regm_test.error.answer

def read_verify_model_flags():
    parser = argparse.ArgumentParser('Model Verify Parameters Parser')

    # General parameters
    parser.add_argument('--predition_device', '-pd', default='1080ti', 
                    type=str, help='predition device for training or testing')
    parser.add_argument('--network_name', '-n', default='perfnetA',
                    type=str, choices=get_nn_list(), help='network name for training or testing')
    parser.add_argument('--loss_name', '-lf', default='maple',
                    type=str, choices=['msle', 'mse', 'maple', 'poisson', 'mae'], help='loss function name for training or testing')
    parser.add_argument('--batch_size', '-b', type=int, default=1, choices=get_support_batchsize(), help='batch size you want to run')
    parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
    parser.add_argument('--iter_benchmark', type=int, default=10, help='Number of iterations for benchmark')
    
    parser.add_argument('--is_train', '-is_train', type=int, default=0, help='collect optimizer data')
    parser.add_argument('--opt_name', '-opt_name', default='sgd',
                    type=str, choices=get_support_optimizer(), help='optimizer name for training or testing')
    
    #Data path Parameters
    parser.add_argument('--data_dirname', '-dd', default='data_full_model', 
        type=str, help='data dirname')
    parser.add_argument('--data_path', '-dp', default='', 
        type=str, help='data path')
   

    #Transformation Parameters
    parser.add_argument('--target_transform', '-tt', default='',
                    type=str, choices=['', 'log'], help='transofrmation for targets')
    parser.add_argument('--feature_transform', '-ft', default='',
                    type=str, choices=['', 'log', 'boxcox', 'sqrt'], help='transofrmation for features')
    parser.add_argument('--ft_list', '-ftl', default=['elements_matrix', 'elements_kernel'],
                    type=str, nargs='+', help='list of features needed to do feature transformations')
    parser.add_argument('--poly', '-poly', default=1, 
                        type=int, help='polynormial for the features')
    parser.add_argument('--magic_scaler', '-magic_scaler', default=1, 
                        type=int, help='magic_scaler for smoothing poisson reg prediction')
 
    #Model and model path Parameters  
    parser.add_argument('--model_dirname', '-md', default='model', 
        type=str, help='model dirname')
    parser.add_argument('--model_path', '-mp', default='', 
        type=str, help='model path')
    
    parser.add_argument('--accuracy_metric', '-am', default='mape', type=str, metavar='PATH',
                       choices=get_support_accuracy_metrics(), help='best accuracy metric for loading the model')


    #Feature and Target filename Parameters
    parser.add_argument('--ft_filename', '-ftfname', 
        default='', type=str, help='Feature and Target file name')
    parser.add_argument('--ft_dirname', '-ftdname', 
        default='Feature_Target', 
        type=str, help='Feature and Target dir name')
    parser.add_argument('--ft_filepath', '-ftf', 
        default='', 
        type=str, help='Feature and Target full path name')

    ##### TBDDDDD 
    parser.add_argument('--gen_model_csv', '-gmc', action="store_true", default=False, help='generate model csv file')
    parser.add_argument('--exe_model', '-em', action="store_true", default=False, help='execute model from csv file')
    parser.add_argument('--profile_model', '-pm', action="store_true", default=False, help='profile model from csv file')
    parser.add_argument('--combine_model', '-cm', action="store_true", default=False, help='combile model from csv file')
    parser.add_argument('--predict_model', '-pdm', action="store_true", default=False, help='predict model from csv file')
    
    # Generate model csv file parameters
    parser.add_argument('--model_name', '-mn',default='lenet', type=str, choices=get_support_models(), help='Neural networks models')
    parser.add_argument('--output_model_csv_filename', '-omcf', type=str, default='', help='The output model csv file name')
    parser.add_argument('--output_model_csv_dirname', '-omcd', type=str, default='model_csv', help='The dirname of the output model csv filename in generation model data step')
    parser.add_argument('--output_model_csv_path', '-omcp', type=str, default='', help='The path of the output model csv filename in generation model data step')
    parser.add_argument('--input_model_csv_filename', '-imcf', type=str, default='', help='The input params csv file from generation data step')
    parser.add_argument('--input_model_csv_dirname', '-imcd', type=str, default='model_csv', help='The dirname of the output csv filename in generation data step')
    parser.add_argument('--input_model_csv_path', '-imcp', type=str, default='', help='The path of the output csv filename in generation data step')
    
    # Execute full model parameters
    parser.add_argument('--output_model_exe_filename', '-omef', type=str, default='', help='The output csv filename in executing data step')
    parser.add_argument('--output_model_exe_dirname', '-omed', type=str, default='model_exe', help='The dirname of the output csv filename in executing data step')
    parser.add_argument('--output_model_exe_path', '-omep', type=str, default='', help='The path of the output csv filename in executing data step')

    # Profile model parameters
    parser.add_argument('--output_timeline_profile_dirname', '-otpd', type=str, default='model_profile', help='The dirname of timeline path')
    parser.add_argument('--output_timeline_profile_path', '-otpp', type=str, default='', help='The timeline path')
    parser.add_argument('--output_timeline_csv_filename', '-otcf', type=str, default='', help='The output model csv file name')
    parser.add_argument('--output_timeline_csv_path', '-otcp', type=str, default='', help='The path of the output model csv filename in generation model data step')
    parser.add_argument('--sleep_time', '-slt', type=float, default=0.005, help='The sleep time for each profile data')

    # Timeline Parser Search Parameters
    parser.add_argument('--replica_cpu', type=str, default='(replica:0)*(CPU:0)+ (Compute)+', help='search tag - replica_cpu')
    parser.add_argument('--all_compute', type=str, default='(GPU:0)*(all Compute)', help='search tag - all_compute')
    parser.add_argument('--memcpy', type=str, default='(memcpy)+ (Compute)+', help='search tag - memcpy')
    parser.add_argument('--compute_transpose_in', type=str, default='TransposeNHWCToNCHW', help='search tag - compute_transpose_in')
    parser.add_argument('--compute_transpose_out', type=str, default='TransposeNCHWToNHWC', help='search tag - compute_transpose_out')    
    parser.add_argument('--memcpyH2D', type=str, default='MEMCPYHtoD', help='search tag - memcpyH2D')
    parser.add_argument('--memcpyD2H', type=str, default='MEMCPYDtoH', help='search tag - memcpyD2H')
    parser.add_argument('--retval', type=str, default='retval', help='search tag - retval')
    parser.add_argument('--first_calculate', type=str, default='sub', help='search tag - first_calculate')
    parser.add_argument('--last_calculate',  type=str, default='Neg', help='search tag - last_calculate')


    # Combile model parameters
    parser.add_argument('--combine_input_params_filename', '-cipf', type=str, default='', help='[Combination] The input csv file name')
    parser.add_argument('--combine_input_params_dirname', '-cipd', type=str, default='model_csv', help='[Combination] The dirname of the input csv filename in generation data step')
    parser.add_argument('--combine_input_params_path', '-cipp', type=str, default='', help='[Combination] The path of the input csv filename in generation data step')
    parser.add_argument('--combine_input_exe_filename', '-cief', type=str, default='', help='[Combination] The input csv filename in executing data step')
    parser.add_argument('--combine_input_exe_dirname', '-cied', type=str, default='model_exe', help='[Combination] The dirname of the input csv filename in executing data step')
    parser.add_argument('--combine_input_exe_path', '-ciep', type=str, default='', help='[Combination] The path of the input csv filename in executing data step')
    parser.add_argument('--combine_input_timeline_profile_filename', '-citpf', type=str, default='', help='[Combination] The input filename of the timeline path')
    parser.add_argument('--combine_input_timeline_profile_dirname', '-citpd', type=str, default='model_profile', help='[Combination] The input dirname of the timeline path')
    parser.add_argument('--combine_input_timeline_profile_path', '-citpp', type=str, default='', help='[Combination] The input path')

    parser.add_argument('--combine_output_filename', '-cof', type=str, default='', help='[Combination] The filename of the output path')
    parser.add_argument('--combine_output_dirname', '-cod', type=str, default='full_csv', help='[Combination] The dirname of the output path')
    parser.add_argument('--combine_output_path', '-cop', type=str, default='', help='[Combination] The output path')


    
    ### Predict model parameters
    parser.add_argument('--train_filename', '-tf', type=str, default='train.csv', help='The input train filename(need all csv to be same)')
    parser.add_argument('--train_path', '-tp', type=str, default='data', help='The input main train path')
    parser.add_argument('--convoulution_sub_path', '-csp', type=str, default='', help='The conv sub train path')
    parser.add_argument('--dense_sub_path', '-dsp', type=str, default='', help='The desne sub train path')
    parser.add_argument('--pooling_sub_path', '-psp', type=str, default='', help='The pooling sub train path')
    parser.add_argument('--output_model_predict_filename', '-ompdf', type=str, default='', help='The output model csv file name')
    parser.add_argument('--output_model_predict_dirname', '-ompdd', type=str, default='model_predict', help='The dirname of the output model csv filename in generation model data step')
    parser.add_argument('--output_model_predict_path', '-ompdp', type=str, default='', help='The path of the output model csv filename in generation model data step')
    

    ### model parameters ### TBD
    parser.add_argument('--start_epoch', default=0, 
                        type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', '-e', default=200, 
                        type=int,  help='number of total epochs to run')
    parser.add_argument('--learning_rate', '-lr', default=0.1, 
                        type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, 
                        type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, 
                        type=float, metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--scheduler_from_json', '-sfj', default='',
                        type=str, help='change scheduler from json file')
    parser.add_argument('--scheduler_mode', '-sched_mode', default='',
                         choices=['',  'fixfreq', 'listrelative'], help='change scheduler mode')
    parser.add_argument('--scheduler_step', '-st', default=150, 
                        type=int, help="step size of scheduler")
    parser.add_argument('--scheduler_gamma', '-sg', default=0.1, 
                        type=float, help="decay rate of scheduler")
                    

    #Feature and Target filename 
    parser.add_argument('--feature_convolution_path', default= os.path.join(os.getcwd(), 'Feature_Target', 'convolution.json'))
    parser.add_argument('--feature_dense_path', default= os.path.join(os.getcwd(), 'Feature_Target', 'dense.json'))
    parser.add_argument('--feature_pooling_path', default= os.path.join(os.getcwd(), 'Feature_Target', 'pooling.json'))
    


    # Testing for memory copy
    parser.add_argument('--memcopyin', '-mci', type=int, default=1, help='the flags for copying data to gpu (0: for generate data in tf GPU)') 

    # Use CPU to run tf model
    parser.add_argument('--cpu', '-cpu',  action='store_true', help='use cpu to run model')
    
    args = parser.parse_args()
    return args

def fullproof_flags(flags):
    flags.model_batch_name = flags.model_name + '_' + str(flags.batch_size)
    flags.model_batch_device_name = flags.model_batch_name + '_' + flags.predition_device
    
    if not flags.data_path:
        flags.data_path = os.path.join(os.getcwd(), flags.data_dirname)
   
    if flags.gen_model_csv:
        if not flags.output_model_csv_filename:
            flags.output_model_csv_filename = flags.model_batch_name + '.csv'
        if not flags.output_model_csv_path:
            flags.output_model_csv_path = os.path.join(flags.data_path, flags.output_model_csv_dirname, flags.output_model_csv_filename)

    if flags.exe_model or flags.profile_model or flags.combine_model or flags.predict_model:
        if not flags.input_model_csv_filename:
            flags.input_model_csv_filename = flags.model_batch_name + '.csv'
        if not flags.input_model_csv_path:
            flags.input_model_csv_path = os.path.join(flags.data_path, flags.input_model_csv_dirname, flags.input_model_csv_filename)
       
    if flags.exe_model:
        if not flags.output_model_exe_filename:
            flags.output_model_exe_filename = flags.model_batch_device_name + '.csv'
        if not flags.output_model_exe_path:
            flags.output_model_exe_path = os.path.join(flags.data_path, flags.output_model_exe_dirname, flags.output_model_exe_filename)

    if flags.profile_model:
        if not flags.output_timeline_profile_path:
            flags.output_timeline_profile_path = os.path.join(flags.data_path, flags.output_timeline_profile_dirname, flags.model_batch_device_name)
        if not flags.output_timeline_csv_filename:
            flags.output_timeline_csv_filename = flags.model_batch_device_name + '.csv'
        if not flags.output_timeline_csv_path:
            flags.output_timeline_csv_path = os.path.join(flags.data_path, flags.output_timeline_profile_dirname, flags.output_timeline_csv_filename)
    
    if flags.combine_model:
        if not flags.combine_input_params_filename:
            flags.combine_input_params_filename = flags.model_batch_name + '.csv'
        if not flags.combine_input_params_path:
            flags.combine_input_params_path = os.path.join(flags.data_path, flags.combine_input_params_dirname, flags.combine_input_params_filename)
        if not flags.combine_input_exe_filename:
            flags.combine_input_exe_filename = flags.model_batch_device_name + '.csv'
        if not flags.combine_input_exe_path:
            flags.combine_input_exe_path = os.path.join(flags.data_path, flags.combine_input_exe_dirname, flags.combine_input_exe_filename)
        if not flags.combine_input_timeline_profile_filename:
            flags.combine_input_timeline_profile_filename = flags.model_batch_device_name + '.csv'
        if not flags.combine_input_timeline_profile_path:
            flags.combine_input_timeline_profile_path = os.path.join(flags.data_path, flags.combine_input_timeline_profile_dirname, flags.combine_input_timeline_profile_filename)

        if not flags.combine_output_filename:
            flags.combine_output_filename = flags.model_batch_device_name + '.csv'
        if not flags.combine_output_path:
            flags.combine_output_path = os.path.join(flags.data_path, flags.combine_output_dirname, flags.combine_output_filename)

    if flags.predict_model:
        flags.network_loss_name = flags.network_name + '_' + flags.loss_name

        if not flags.output_model_predict_filename:
            flags.output_model_predict_filename = flags.model_batch_device_name + '.csv'
        if not flags.output_model_predict_path:
            flags.output_model_predict_path = os.path.join(flags.data_path, flags.output_model_predict_dirname, flags.output_model_predict_filename)
        
        if not flags.convoulution_sub_path:
            flags.convoulution_sub_path = 'convolution_' + flags.predition_device
        if not flags.dense_sub_path:
            flags.dense_sub_path = 'dense_' + flags.predition_device
        if not flags.pooling_sub_path:
            flags.pooling_sub_path = 'pooling_' + flags.predition_device
       
        if flags.accuracy_metric == 'mae':
            flags.amdirname = flags.network_loss_name + '_mae'
        elif flags.accuracy_metric == 'mape':
            flags.amdirname = flags.network_loss_name + '_mape'
        elif flags.accuracy_metric == 'rmse':
            flags.amdirname = flags.network_loss_name + '_rmse'
        else:
            flags.amdirname = flags.network_loss_name + '_r2'
    
    return flags

def auto_create_dir(flags):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    create_dir_elemenet(flags.data_path)
    if flags.gen_model_csv:
        create_dir_elemenet(os.path.dirname(flags.output_model_csv_path))
    if flags.exe_model:
        create_dir_elemenet(os.path.dirname(flags.output_model_exe_path))
    if flags.profile_model:
        create_dir_elemenet(flags.output_timeline_profile_path)
    if flags.combine_model:
        create_dir_elemenet(os.path.dirname(flags.combine_output_path))
    if flags.predict_model:
        create_dir_elemenet(os.path.dirname(flags.output_model_predict_path))
    return

def get_op(layer, index, dict_layernum, memcpyin, inputs, is_train = 0):
    input_ = None
    if re.search('conv2d', layer['operation'], re.M|re.I):
        if memcpyin:
            input_ = np.random.normal(127, 60, (layer['batchsize'].astype(int), layer['matsize'].astype(int), 
                layer['matsize'].astype(int), layer['channels_in'].astype(int))).astype(float)
            inputs = tf.placeholder(tf.float32, shape=[layer['batchsize'].astype(int), layer['matsize'].astype(int), 
                layer['matsize'].astype(int), layer['channels_in'].astype(int)], name="inputs")                    
        op = create_conv2d(inputs, layer, 'conv2d', dict_layernum, is_train = is_train)
        print("Type of layer{} is {}".format(index+1, 'conv2d'))    

    elif re.search('pooling', layer['operation'], re.M | re.I):
        if memcpyin:
            input_ = np.random.normal(127, 60, (layer['batchsize'].astype(int), layer['matsize'].astype(int), 
                layer['matsize'].astype(int), layer['channels_in'].astype(int))).astype(float)
            inputs = tf.placeholder(tf.float32, shape=[layer['batchsize'].astype(int), layer['matsize'].astype(int), 
                layer['matsize'].astype(int), layer['channels_in'].astype(int)], name="inputs")
        op = create_pooling(inputs, layer, 'pooling', dict_layernum, is_train = is_train)
        print("Type of layer{} is {}".format(index+1, 'pooling'))    

    elif re.search('dense', layer['operation'], re.M | re.I):
        if memcpyin:
            input_ = np.random.normal(127, 60, (layer['batchsize'].astype(int), layer['dim_input'].astype(int))).astype(float)
            inputs = tf.placeholder(tf.float32, shape=[layer['batchsize'].astype(int), layer['dim_input'].astype(int)], name="inputs")                    
        op = create_dense(inputs, layer, 'dense', dict_layernum, is_train = is_train)
        print("Type of layer{} is {}".format(index+1, 'dense'))
    return op, inputs, input_

def main():
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    success_tag = colored('[Success] ', 'green')
    flags = read_verify_model_flags()
    flags = fullproof_flags(flags)
    if flags.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(warn_tag + 'Force to use cpu to compute')
    auto_create_dir(flags)

    model_csv_columns = get_model_csv_total_columns(is_train=flags.is_train)
    end_to_end_model  = get_end_to_end_model(flags.model_name)
    #print(model_csv_columns)
    
    if flags.gen_model_csv:
        print("[Generate Model CSV File]")
        if os.path.isfile(flags.output_model_csv_path):
            print(warn_tag, "Alreadly have the data in %s, pass this step!" % flags.output_model_csv_path)
        else:
            layer_count = 0
            df_layers = pd.DataFrame(columns=model_csv_columns, index=None)
            for _, layer in enumerate(end_to_end_model.layers):
                dict_layer  = dict.fromkeys(model_csv_columns)
                founded = 0
                if re.search('conv2d', str(layer.__class__), re.M|re.I):
                    cfg = layer.get_config()
                    input_shape_re = re.findall('[0-9]+', str(layer.input_shape))
                    dict_layer['matsize']      = int(input_shape_re[1])
                    dict_layer['channels_in']  = int(input_shape_re[2])
                    dict_layer['channels_out'] = int(cfg["filters"])
                    dict_layer['kernelsize']   = int(re.findall('[0-9]+', str(cfg["kernel_size"]))[0])
                    dict_layer['strides']      = int(re.findall('[0-9]+', str(cfg["strides"]))[0])
                    dict_layer['padding']      = 1 if cfg["padding"]=='same' else 0
                    dict_layer['use_bias']     = 1 if cfg["use_bias"]==True  else 0
                    dict_layer['activation_fct']  = 1 if cfg["activation"] != None else 0
                    dict_layer['elements_matrix'] = dict_layer['matsize'] ** 2
                    dict_layer['elements_kernel'] = dict_layer['kernelsize'] ** 2
                    founded = 1        
                elif re.search('MaxPooling2D', str(layer.__class__), re.M|re.I):
                    cfg = layer.get_config()
                    input_shape_re = re.findall('[0-9]+', str(layer.input_shape))
                    dict_layer['matsize']      = int(input_shape_re[1])
                    dict_layer['channels_in']  = int(input_shape_re[2])
                    dict_layer['poolsize']     = int(re.findall('[0-9]+', str(cfg["pool_size"]))[0])
                    dict_layer['strides']      = int(re.findall('[0-9]+', str(cfg["strides"]))[0])
                    dict_layer['padding']      = 1 if cfg["padding"]=='same' else 0
                    dict_layer['elements_matrix'] = dict_layer['matsize'] ** 2
                    founded = 1
                elif re.search('GlobalAveragePooling2D', str(layer.__class__), re.M|re.I):
                    input_shape_re = re.findall('[0-9]+', str(layer.input_shape))
                    dict_layer['matsize']      = int(input_shape_re[1])
                    dict_layer['channels_in']  = int(input_shape_re[2])
                    dict_layer['poolsize']     = dict_layer['matsize']
                    dict_layer['strides']      = 1
                    dict_layer['padding']      = 0
                    dict_layer['elements_matrix'] = dict_layer['matsize'] ** 2
                    founded = 1
                elif re.search('Dense', str(layer.__class__), re.M|re.I):
                    cfg = layer.get_config()
                    input_shape_re = re.findall('[0-9]+', str(layer.input_shape))
                    dict_layer['matsize']        = 1
                    dict_layer['activation_fct'] = 1 if cfg["activation"] != None else 0
                    dict_layer['dim_input']      = int(input_shape_re[0])
                    dict_layer['dim_output']     = int(cfg["units"])
                    founded = 1
                if flags.is_train:
                    # reset the opt 
                    dict_layer['sgd']     = 0
                    dict_layer['adagrad'] = 0
                    dict_layer['rmsprop'] = 0
                    dict_layer['adam']    = 0
                    # Correct the opt 
                    dict_layer[flags.opt_name] = 1
                if founded:
                    # Basic params
                    dict_layer['name']      = layer.name
                    dict_layer['layers']    = layer_count + 1
                    dict_layer['operation'] = str(layer.__class__)
                    # TBD - activation-fct detail setting 
                    '''
                    next_index = index + 1
                    if next_index <= len(model.layers) and re.search('Activation', str(model.layers[next_index].__class__), re.M|re.I):
                        dict_layer['activation_fct'] = 1 if cfg["activation"] != None else 0
                    '''
                    # Add this row to dataframe
                    df_layer  = pd.DataFrame(dict_layer, columns=model_csv_columns, index=[0])
                    df_layers = df_layers.append(df_layer) 
                    layer_count += 1

            df_layers['batchsize'] = flags.batch_size
            df_layers.to_csv(flags.output_model_csv_path, index=False)
            print(success_tag, "Create file to %s!" % flags.output_model_csv_path)

    if flags.exe_model or flags.profile_model or flags.combine_model or flags.predict_model:
        if not os.path.isfile(flags.input_model_csv_path):
            print(flags.input_model_csv_path)
            print(warn_tag, "Please create model csv file or open the '-gmc' tag for the supported model at first!")
            return
    if flags.exe_model:
        dict_layernum = {'conv2d': 0, 'pooling': 0, 'dense': 0}
        df_ = pd.read_csv(flags.input_model_csv_path, usecols=model_csv_columns)
        tf.reset_default_graph()
        for index in range(df_.shape[0]):
            tf.reset_default_graph()
            time_list = []
            layer = df_.loc[index, :]
            inputs = None 
            op, inputs, input_ = get_op(layer, index, dict_layernum, flags.memcopyin, inputs, is_train = flags.is_train)
        
            sess = tf.Session()
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            
            if flags.memcopyin:
                # Do WarmUp               
                for _ in range(flags.iter_warmup):
                    sess.run(op, feed_dict = {inputs: input_})
                # Do Benchmark
                for _ in range(flags.iter_benchmark):
                    start_time = time.time()
                    sess.run(op, feed_dict = {inputs: input_})
                    time_list.append(((time.time()-start_time) * 1000)) # The unit of measurement is 'Millisecond'
            else:
                # Warm-up run
                for _ in range(flags.iter_warmup):
                    sess.run(op)
                # Benchmark run
                for _ in range(flags.iter_benchmark):
                    start_time = time.time()
                    sess.run(op)
                    time_list.append(((time.time()-start_time) * 1000))
                    
            np_array_parameters = np.array(time_list)
            df_.loc[index, 'batchsize'] = flags.batch_size
            df_.loc[index, 'time_max']  = np.amax(np_array_parameters)
            df_.loc[index, 'time_min']  = np.amin(np_array_parameters)
            df_.loc[index, 'time_median']    = np.median(np_array_parameters)
            df_.loc[index, 'time_mean']      = np.mean(np_array_parameters)
            df_.loc[index, 'time_trim_mean'] = stats.trim_mean(np_array_parameters, 0.1)
        df_.to_csv(flags.output_model_exe_path, index=False)

    if flags.profile_model:
        dict_layernum = {'conv2d': 0, 'pooling': 0, 'dense': 0}
        df_ = pd.read_csv(flags.input_model_csv_path, usecols=model_csv_columns)
        tf.reset_default_graph()
        run_metadata = tf.RunMetadata()
        for index in range(df_.shape[0]):
            tf.reset_default_graph()
            filename = os.path.join(flags.output_timeline_profile_path, 'layer_' + str(index+1) + '.json')
            if os.path.isfile(filename):
                print(warn_tag, "File: {} is Existed, Pleas delete all duplicate files manually!".format((filename)))
                break

            layer = df_.loc[index, :]
            inputs = None 
            op, inputs, input_ = get_op(layer, index, dict_layernum, flags.memcopyin, inputs, is_train = flags.is_train)

            sess = tf.Session()
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            
            if flags.memcopyin:
                # Do WarmUp
                for _ in range(flags.iter_warmup):
                    sess.run(op, feed_dict = {inputs: input_})
                # Do Benchmark            
                start_time = time.time()
                sess.run(op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata, feed_dict = {inputs: input_})
                print(((time.time()-start_time) * 1000), "ms") # The unit of measurement is 'Millisecond'

            else:
                # Do WarmUp
                for _ in range(flags.iter_warmup):
                    sess.run(op)
                # Do Benchmark            
                sess.run(op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)

            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(filename, 'w') as f:
                f.write(ctf) 
            time.sleep(flags.sleep_time)
        
        files_input_timeline_profile_path = os.listdir(flags.output_timeline_profile_path)
        # print(all_files)
        index = 0
        for filename in files_input_timeline_profile_path:
            full_filename = os.path.join(flags.output_timeline_profile_path, filename)
            '''
            recorders = Recorders(json_filename = full_filename, json_data = None, 
                str_replica_cpu = flags.replica_cpu, str_replica_gpu = flags.replica_gpu, 
                str_all_compute = flags.all_compute, str_memcpy = flags.memcpy,
                str_replica_transpose_in  = flags.replica_transpose_in, 
                str_replica_transpose_out = flags.replica_transpose_out, 
                str_compute_transpose_in  = flags.compute_transpose_in, 
                str_compute_transpose_out = flags.compute_transpose_out,
                str_memcpyH2D = flags.memcpyH2D, str_memcpyD2H = flags.memcpyD2H, 
                str_retval =flags.retval)
            date_ele = {
                'layers':             str(os.path.splitext(filename)[0]).split('_')[1],
                'preprocess_time':    recorders.preprocess_time,
                'execution_time':     recorders.execution_time,
                'memcpy_time':        recorders.memcpyD2H_time, 
                'retval_time':        recorders.retval_time,
                'retval_half_time':   recorders.retval_half_time,
                'memcpy_retval':      recorders.memcpyD2H_time + recorders.retval_time,
                'memcpy_retval_half': recorders.memcpyD2H_time + recorders.retval_half_time,
                'sess_time':          recorders.sess_time
            }
            for key, value in date_ele.items():
                if key == 'layers':
                    continue
                date_ele[key] = value / 1000 # The unit of measurement is 'Millisecond' 
            df_ele = pd.DataFrame(data = date_ele, index=[0])
            '''
            recorders = Recorders(json_filename = full_filename, json_data = None, 
                str_replica_cpu = flags.replica_cpu,
                str_all_compute = flags.all_compute, str_memcpy = flags.memcpy,
                str_compute_transpose_in  = flags.compute_transpose_in, 
                str_compute_transpose_out = flags.compute_transpose_out,
                str_memcpyH2D = flags.memcpyH2D, str_memcpyD2H = flags.memcpyD2H, 
                str_retval = flags.retval, 
                str_first_calculate= flags.first_calculate,
                str_last_calculate = flags.last_calculate,
                is_train = flags.is_train)
            
            for key, value in recorders.data.items():
                if key == 'hashkey':
                    continue
                recorders.data[key] = value / 1000 # The unit of measurement is 'Millisecond' 
            df_ele = pd.DataFrame(data = recorders.data, index=[0])
            
            if index==0: 
                df_ele.to_csv(flags.output_timeline_csv_path, index=False)
                index = 1
            else:
                df_ele.to_csv(flags.output_timeline_csv_path, index=False, mode='a', header=False)

    if flags.combine_model:
        if os.path.isfile(flags.combine_output_path):
            print(warn_tag, "File: {} is Existed, Pleas delete it manually!".format((flags.combine_output_path)))
        else:
            col_dict  = get_colnames_from_dict()
            df_struct = pd.read_csv(flags.combine_input_params_path)
            df_all = df_struct.copy()
            if os.path.isfile(flags.combine_input_exe_path):
                df_time     = pd.read_csv(flags.combine_input_exe_path)
                for ct in col_dict['time']:
                    df_all[ct] = df_all.layers.map(df_time.set_index('layers')[ct])
            else:
                print(warn_tag, "No time data is also acceptable for convience!")
            
            if not flags.cpu: ## 不是cpu才有timeline可以合併
                df_timeline = pd.read_csv(flags.combine_input_timeline_profile_path)
                for ct in col_dict['profile']:
                    df_all[ct] = df_all.layers.map(df_timeline.set_index('layers')[ct])
            df_all.to_csv(flags.combine_output_path, index=False)

    if flags.predict_model:
        print("[Predict Model]")
        if os.path.isfile(flags.output_model_predict_path):
            print(warn_tag, "Alreadly have the data in %s, pass this step!" % flags.output_model_predict_path)
            df_  = pd.read_csv(flags.input_model_csv_path)
            return
        feature_conv, _    = get_feature_target(flags.feature_convolution_path) # get the feature and target 
        feature_pooling, _ = get_feature_target(flags.feature_pooling_path) # get the feature and target 
        feature_dense, _   = get_feature_target(flags.feature_dense_path) # get the feature and target 
        df_  = pd.read_csv(flags.input_model_csv_path)
        target_list = ['time_mean', 'preprocess_time', 'execution_time', 'memcpy_retval_half']
        dict_target ={
            'time_mean': '',
            'preprocess_time' : 'pre',
            'execution_time' : 'exe',
            'memcpy_retval_half' : 'post'
        }
        dict_target_list ={
            'time_mean': [],
            'preprocess_time' : [],
            'execution_time' : [],
            'memcpy_retval_half' : []
        }
        for index in range(df_.shape[0]):
            print(' ======== %d ========' % index)
            layer = df_.loc[index, :]
            if re.search('conv2d', layer['operation'], re.M|re.I):
                df_test = layer.fillna(0)
                for target in target_list:
                    if index > 0 and target == 'preprocess_time':
                        dict_target_list[target].append(0)
                        continue
                    if index != (df_.shape[0]-1) and target == 'memcpy_retval_half':
                        dict_target_list[target].append(0)
                        continue
                    df_test = layer.fillna(0)
                    df_test = pd.DataFrame(df_test.values.reshape(-1, len(layer)), columns=df_.columns)     
                    df_test[feature_conv] = df_test[feature_conv].astype(int)
                    flags.batch_size = 128
                    model = Model(flags, len(feature_conv))

                    flags.train_csv = os.path.join(os.getcwd(), flags.train_path, flags.convoulution_sub_path, flags.train_filename)
                    flags.ft_list   = ['elements_matrix', 'elements_kernel']
                    train_feature, train_target, test_feature, test_target = pred_data_preparation(flags, df_test, feature_conv, target)
                    print("----------->", dict_target[target], test_feature)
                    if not dict_target[target]:
                        if not train_feature:
                            continue
                        flags.data_ex_basename = 'convolution_' + flags.predition_device 
                    else:
                        flags.data_ex_basename = 'convolution_' + dict_target[target] + '_' + flags.predition_device 
                    ckpt_path_name = os.path.join(os.getcwd(), flags.model_dirname, flags.data_ex_basename, flags.amdirname)
                    print(ckpt_path_name)
                    pred_ele, anw_t  = pred_validation(flags, model, ckpt_path_name, test_feature, test_target)                    
                    dict_target_list[target].append(pred_ele[0])
                    print(dict_target_list)
            

            elif re.search('pooling2d', layer['operation'], re.M|re.I):
                df_test = layer.fillna(0)
                for target in target_list:
                    if index > 0 and target == 'preprocess_time':
                        dict_target_list[target].append(0)
                        continue
                    if index != (df_.shape[0]-1) and target == 'memcpy_retval_half':
                        dict_target_list[target].append(0)
                        continue
                    df_test = layer.fillna(0)
                    df_test = pd.DataFrame(df_test.values.reshape(-1, len(layer)), columns=df_.columns)     
                    df_test[feature_pooling] = df_test[feature_pooling].astype(int)
                    flags.batch_size = 128
                    model = Model(flags, len(feature_pooling))
                    flags.train_csv = os.path.join(os.getcwd(), flags.train_path, flags.pooling_sub_path, flags.train_filename)
                    flags.ft_list   = ['elements_matrix']
                    train_feature, train_target, test_feature, test_target = pred_data_preparation(flags, df_test, feature_pooling, target)
                    print("----------->", dict_target[target])
                    if not dict_target[target]:
                        if not train_feature:
                            continue
                        flags.data_ex_basename = 'pooling_' + flags.predition_device 
                    else:
                        flags.data_ex_basename = 'pooling_' + dict_target[target] + '_' + flags.predition_device 
                    ckpt_path_name = os.path.join(os.getcwd(), flags.model_dirname, flags.data_ex_basename, flags.amdirname)

                    pred_ele, anw_t  = pred_validation(flags, model, ckpt_path_name, test_feature, test_target)                    
                    dict_target_list[target].append(pred_ele[0])
                    print(dict_target_list)
             
            elif re.search('dense', layer['operation'], re.M|re.I):
                df_test = layer.fillna(0)
                for target in target_list:
                    if index > 0 and target == 'preprocess_time':
                        dict_target_list[target].append(0)
                        continue
                    if index != (df_.shape[0]-1) and target == 'memcpy_retval_half':
                        dict_target_list[target].append(0)
                        continue
                    df_test = layer.fillna(0)
                    df_test = pd.DataFrame(df_test.values.reshape(-1, len(layer)), columns=df_.columns)

                    df_test[feature_dense] = df_test[feature_dense].astype(int)
                    flags.batch_size = 128
                    model = Model(flags, len(feature_dense))
                    flags.train_csv = os.path.join(os.getcwd(), flags.train_path, flags.dense_sub_path, flags.train_filename)
                    flags.ft_list = ""
                    if target == 'memcpy_retval_half':
                        flags.magic_scaler = 10
                    train_feature, train_target, test_feature, test_target = pred_data_preparation(flags, df_test, feature_dense, target)

                    print("----------->", dict_target[target])
                    if not dict_target[target]:
                        if not train_feature:
                            continue
                        flags.data_ex_basename = 'dense_' + flags.predition_device 
                    else:
                        flags.data_ex_basename = 'dense_' + dict_target[target] + '_' + flags.predition_device 
                    ckpt_path_name = os.path.join(os.getcwd(), flags.model_dirname, flags.data_ex_basename, flags.amdirname)

                    pred_ele, anw_t  = pred_validation(flags, model, ckpt_path_name, test_feature, test_target)                    
                    dict_target_list[target].append(pred_ele[0])
                    print(dict_target_list)

             
        if dict_target_list['preprocess_time']:
            df_['pred_pre_time'] = dict_target_list['preprocess_time']
        if dict_target_list['execution_time']:
            df_['pred_exe_time'] = dict_target_list['execution_time']
        if dict_target_list['memcpy_retval_half']:
            df_['pred_post_time'] = dict_target_list['memcpy_retval_half']
        if dict_target_list['time_mean']:
            df_['pred_sess_time'] = dict_target_list['time_mean']
        if dict_target_list['preprocess_time'] and  dict_target_list['execution_time'] and dict_target_list['memcpy_retval_half']:
            sum_time = np.sum(df_['pred_exe_time']) + df_.iloc[0]['pred_pre_time'] + df_.iloc[-1]['pred_post_time']
            df_['sum_time'] = sum_time
        df_.to_csv(flags.output_model_predict_path, index=False)
        
        print(success_tag, "Create file to %s!" % flags.output_model_predict_path)

if __name__ == '__main__':
    main()
