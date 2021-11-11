import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
#import tensorflow as tf

from scipy import stats
from termcolor import colored
#from tensorflow.python.client import timeline

from utils.utils import get_support_layers, get_support_devices
from utils.parameters import get_params 
from utils.layers import *

def read_collect_data_flags():
    parser = argparse.ArgumentParser('Data Collection')
    # General Parameters
    parser.add_argument('--predition_layertype', '-pl', 
        default='convolution', type=str, choices = get_support_layers(), 
        help='The layer types of the neural networks') ### TBD: Chaning this tag name 
    parser.add_argument('--device', '-d', type=str, 
        default='1080ti', help='The target device Name for benchmarking data')
    parser.add_argument('--num', '-num', type=int, 
        default=110000, help='The number of results to compute')
    parser.add_argument('--shuffle', '-shuffle', action="store_true", 
        default=True, help='shuffle the data')
    
    parser.add_argument('--run_on_child', '-roc', type=int, 
        default=1, help='run on child process')
    
    parser.add_argument('--memory_monitor', '-mm',  type=int, 
        default=1, help='run on child process with memory monitor')


    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument('--gen_params', '-gp', action="store_true", 
        default=False, help='generate random parameters')
    group1.add_argument('--exe_params', '-ep', action="store_true", 
        default=False, help='execution parameters on specific device')
    group1.add_argument('--profile_params', '-pp', action="store_true", 
        default=False, help='Use tensorflow profiler')
    
    group1.add_argument('--trt_freeze', '-trtf', action="store_true", 
        default=False, help='tensorflowRT freeze graph to pb')
    group1.add_argument('--trt_exe', '-trtexe', action="store_true", 
        default=False, help='use tensorflowRT graph and execute its pb')
    group1.add_argument('--trt_profile', '-trtp', action="store_true", 
        default=False, help='use tensorflowRT graph and execute its pb with profiler')
    
    parser.add_argument('--path', '-path', type=str, 
        default='', help='The main path of this program')
    parser.add_argument('--is_train', '-is_train', type=int, 
        default=0, help='collect optimizer data')
        
    # Generate Parameters
    parser.add_argument('--output_params_filename', '-opf', type=str, 
        default='', help='The output csv file name')
    parser.add_argument('--output_params_dirname', '-opd', type=str, 
        default='struct', help='The dirname of the output csv filename in generation data step')
    parser.add_argument('--output_params_path', '-opp', type=str, 
        default='', help='The path of the output csv filename in generation data step')
     
    # Not-Generate Parameters
    parser.add_argument('--input_params_filename', '-ipf', type=str, 
        default='', help='The input params csv file from generation data step')
    parser.add_argument('--input_params_dirname', '-ipd', type=str, 
        default='struct', help='The dirname of the output csv filename in generation data step')
    parser.add_argument('--input_params_path', '-ipp', type=str, 
        default='', help='The path of the output csv filename in generation data step')

    # Execution Parameters
    parser.add_argument('--output_exe_filename', '-oef', type=str, 
        default='', help='The output csv filename in executing data step')
    parser.add_argument('--output_exe_dirname', '-oed', type=str, 
        default='time', help='The dirname of the output csv filename in executing data step')
    parser.add_argument('--output_exe_path', '-oep', type=str,
        default='', help='The path of the output csv filename in executing data step')
    parser.add_argument('--iter_warmup', type=int, 
        default=5, help='Number of iterations for warm-up')
    parser.add_argument('--iter_benchmark', '-b', type=int, 
        default=10, help='Number of iterations for benchmark')
    parser.add_argument('--cpu', '-cpu', action="store_true", 
        default=False, help='Force to use CPU to computate')

    # Profile Parameters
    parser.add_argument('--output_timeline_profile_dirname', '-otpd', type=str, 
        default='timeline_profile', help='The dirname of timeline path')
    parser.add_argument('--output_timeline_profile_path', '-otpp', type=str, 
        default='', help='The timeline path')
    parser.add_argument('--sleep_time', '-st', type=float, 
        default=0.005, help='The sleep time for each profile data')
    # parser.add_argument('--backup_path', type=str, default=os.path.join(os.getcwd(), 'backup'), help='backup path')
    
    ### New Feature: TensorRT
    # TensorRT pb file Parameters
    parser.add_argument('--trt_frozen_pb_dir', '-trtfpd', type=str, 
        default='trt_pb', help='The dirname of tensorRT pb file path')
    parser.add_argument('--trt_frozen_pb_path', '-trtfpp', type=str, 
        default='', help='The path of tensorRT pb file path')
    # TensorRT pb file Parameters
    parser.add_argument('--trt_frozen_json_dir', '-trtfjd', type=str, 
        default='trt_json', help='The dirname of tensorRT pb file path')
    parser.add_argument('--trt_frozen_json_path', '-trtfjp', type=str, 
        default='', help='The path of tensorRT pb file path')
    
    # TensorRT Execution Parameters
    parser.add_argument('--trt_output_exe_filename', '-trtoef', type=str, 
        default='', help='The output csv filename in tensorRT executing data step')
    parser.add_argument('--trt_output_exe_dirname', '-trtoed', type=str, 
        default='trt_time', help='The dirname of the output csv filename in tensorRT executing data step')
    parser.add_argument('--trt_output_exe_path', '-trtoep', type=str,
        default='', help='The path of the output csv filename in tensorRT executing data step')

    # TensorRT Profile Parameters
    parser.add_argument('--trt_output_timeline_profile_dirname', '-trtotpd', type=str, 
        default='trt_timeline_profile', help='The dirname of tensorRT timeline path')
    parser.add_argument('--trt_output_timeline_profile_path', '-trtotpp', type=str, 
        default='', help='The tensorRT timeline path')

    # Testing for memory copy
    parser.add_argument('--memory_copy_in', '-mci', type=int, 
        default=1, help='the flags for copying data to gpu (0: for generate data in tf GPU)') 
    args = parser.parse_args()
    return args

def fullproof_flags(flags):
    default_dirname = 'data'
    if flags.is_train:
        default_dirname = 'train_data'
    if not flags.path:
        flags.path = os.path.join(os.getcwd(), default_dirname)
            
    flags.data_basename = flags.predition_layertype + '_' + flags.device
    str_param_csv = '_parameters.csv'
    if flags.gen_params:
        if not flags.output_params_filename:
            flags.output_params_filename = flags.predition_layertype + str_param_csv
        if not flags.output_params_path:
            flags.output_params_path = os.path.join(flags.path, flags.output_params_dirname, flags.output_params_filename)
    
    if flags.exe_params or flags.profile_params:
        if not flags.input_params_filename:
            flags.input_params_filename = flags.predition_layertype + str_param_csv
        if not flags.input_params_path:
            flags.input_params_path = os.path.join(flags.path, flags.input_params_dirname, flags.input_params_filename)

    if flags.exe_params:
        if not flags.output_exe_filename:
            flags.output_exe_filename = flags.data_basename + '.csv'
        if not flags.output_exe_path:
            flags.output_exe_path = os.path.join(flags.path, flags.output_exe_dirname, flags.output_exe_filename)
    
    if flags.profile_params:
        if not flags.output_timeline_profile_path:
            flags.output_timeline_profile_path = os.path.join(flags.path, flags.output_timeline_profile_dirname, flags.data_basename)
    
    ### New Feature: TensorRT
    if flags.trt_freeze or flags.trt_exe or flags.trt_profile:
        if not flags.input_params_filename:
            flags.input_params_filename = flags.predition_layertype + str_param_csv
        if not flags.input_params_path:
            flags.input_params_path = os.path.join(flags.path, flags.input_params_dirname, flags.input_params_filename)
        
        if not flags.trt_frozen_pb_path:
            #flags.trt_frozen_pb_dir = flags.trt_frozen_pb_dir + 'mci1' if flags.memory_copy_in else \
            #        flags.trt_frozen_pb_dir + 'mci0' ### TBD mci 1 only!!
            flags.trt_frozen_pb_path = os.path.join(flags.path, flags.trt_frozen_pb_dir, flags.predition_layertype)
        
        if not flags.trt_frozen_json_path:
            flags.trt_frozen_json_path = os.path.join(flags.path, flags.trt_frozen_json_dir, flags.predition_layertype)
    
    if flags.trt_exe:
        if not flags.trt_output_exe_filename:
            flags.trt_output_exe_filename = flags.data_basename + '.csv'
        if not flags.trt_output_exe_path:
            flags.trt_output_exe_path = os.path.join(flags.path, flags.trt_output_exe_dirname, flags.trt_output_exe_filename)
    
    if flags.trt_profile:
        if not flags.trt_output_timeline_profile_path:
            flags.trt_output_timeline_profile_path = os.path.join(flags.path, flags.trt_output_timeline_profile_dirname, flags.data_basename)        
    return flags

def auto_create_dir(flags):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    create_dir_elemenet(flags.path)
    if flags.gen_params:
        create_dir_elemenet(os.path.dirname(flags.output_params_path))
    if flags.exe_params:
        create_dir_elemenet(os.path.dirname(flags.output_exe_path))
    if flags.profile_params:
        create_dir_elemenet(flags.output_timeline_profile_path)
    ### New Feature: TensorRT
    if flags.trt_freeze or flags.trt_exe or flags.trt_profile:
        create_dir_elemenet(flags.trt_frozen_pb_path)
        create_dir_elemenet(flags.trt_frozen_json_path)

    if flags.trt_exe:
        create_dir_elemenet(os.path.dirname(flags.trt_output_exe_path))
    if flags.trt_profile:
        create_dir_elemenet(flags.trt_output_timeline_profile_path)
    return

def check_config(flags):
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    success_tag = colored('[Success] ', 'green')
    device_dict = get_support_devices()
    if flags.device in device_dict.keys():
        foolproof_device = device_dict[flags.device]
        if foolproof_device.lower() == 'cpu':
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(success_tag + 'foolproof: Use ' + foolproof_device + ' to compute')
    if flags.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(warn_tag + 'Force to use cpu to compuate')
    return

def output_path_setting(flags):
    output_path = None
    if flags.exe_params:
        print("[Execution]")
        if os.path.isfile(flags.output_exe_path):
            print(colored('[Warn] ', 'red'), 
                "File: {} is Existed, Pleas delete it manually!".format((flags.output_exe_path)))
            exit()
        output_path = flags.output_exe_path
    elif flags.profile_params:
        print("[Profiling]")
        output_path = flags.output_timeline_profile_path
    elif flags.trt_freeze:
        print("[TRT Freezing]")
    elif flags.trt_exe:
        print("[TRT Execution]")
        if os.path.isfile(flags.trt_output_exe_path):
            print(colored('[Warn] ', 'red'), 
                "File: {} is Existed, Pleas delete it manually!".format((flags.trt_output_exe_path)))
            exit()
        output_path = flags.trt_output_exe_path
    elif flags.trt_profile:
        output_path = flags.trt_output_timeline_profile_path
        print("[TRT Profiling]")
    else:
        print("This type of mode is not support!")
        exit()
    return output_path

def main():
    success_tag = colored('[Success] ', 'green')
    warn_tag    = colored('[Warn] ', 'red') 
    flags = read_collect_data_flags()
    flags = fullproof_flags(flags)
    print(flags)
    auto_create_dir(flags)

    if flags.gen_params:
        print("[Generate Random data]")
        if os.path.isfile(flags.output_params_path):
            print(success_tag, "Alreadly have the dataset, pass this step!")
        else:
            print(warn_tag, "Create random dataset to", flags.output_params_path)
            params = get_params(flags.num, flags.predition_layertype, is_train = flags.is_train)
            if params:
                params.generate_params_with_hashkey()
                df_data = params.get_shuffle_data() if flags.shuffle else params.data        
                df_data.to_csv(flags.output_params_path, index=False)
                del df_data
        print(success_tag, 'Generate Random data is Done!')

    else:
        if not os.path.isfile(flags.input_params_path):
            ##prepare data from input csv
            print(warn_tag, "Please create random dataset or open the '-gp' tag for the supported layer at first!")
            return
        check_config(flags)
        df_data  = pd.read_csv(flags.input_params_path)
        params   = get_params(df_data.shape[0], flags.predition_layertype)
        params.set_data(df_data)
        params.auto_generate_elements()
        params.generate_hashkey()
        output_path = output_path_setting(flags)
        
        for index, row in params.data.iterrows():
            print("============  {}  ===========".format(index+1))
            if index == flags.num:
                break
            
            if flags.exe_params or flags.profile_params:
                layer = get_layer(flags, row, output_path)
                if flags.run_on_child:
                    layer.run_on_child(0, flags.exe_params, 
                        flags.profile_params, flags.memory_monitor)    
                else:
                    layer.run(flags.exe_params, flags.profile_params)

            elif flags.trt_freeze or flags.trt_exe or flags.trt_profile:
                layer = get_trt_layer(flags, row, output_path)
                if flags.run_on_child:
                    layer.run_on_child(flags.trt_freeze, flags.trt_exe, 
                        flags.trt_profile, flags.memory_monitor)
                else:
                    layer.run(flags.trt_exe, flags.trt_profile)
    
if __name__ == '__main__':
    main()
