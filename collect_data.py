import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import stats
from termcolor import colored
from tensorflow.python.client import timeline

from utils.utils import get_support_layers, get_support_devices, get_colnames, write_file
from utils.parameters import get_params 

def read_collect_data_flags():
    parser = argparse.ArgumentParser('Data Collection')
    # General Parameters
    parser.add_argument('--predition_layertype', '-pl', default='convolution', type=str, choices=get_support_layers(), help='Layer types of neural networks')
    parser.add_argument('--device', '-d', type=str, default='1080ti', help='target device Name for benchmarking')
    parser.add_argument('--gen_params', '-gp', action="store_true", default=False, help='generate random parameters')
    parser.add_argument('--exe_params', '-ep', action="store_true", default=False, help='execution parameters on specific device')
    parser.add_argument('--profile_params', '-pp', action="store_true", default=False, help='Use tensorflow profiler')
    parser.add_argument('--is_train', '-is_train', type=int, default=0, help='collect optimizer data')
    parser.add_argument('--path', '-path', type=str, default='', help='The main path of this program')

    # Generate Parameters
    parser.add_argument('--num', '-num', type=int, default=110000, help='Number of results to compute')
    parser.add_argument('--shuffle', '-shuffle', action="store_true", default=True, help='shuffle the data')
    parser.add_argument('--output_params_filename', '-opf', type=str, default='', help='The output csv file name')
    parser.add_argument('--output_params_dirname', '-opd', type=str, default='struct', help='The dirname of the output csv filename in generation data step')
    parser.add_argument('--output_params_path', '-opp', type=str, default='', help='The path of the output csv filename in generation data step')
    
    # Execution Parameters
    parser.add_argument('--input_params_filename', '-ipf', type=str, default='', help='The input params csv file from generation data step')
    parser.add_argument('--input_params_dirname', '-ipd', type=str, default='struct', help='The dirname of the output csv filename in generation data step')
    parser.add_argument('--input_params_path', '-ipp', type=str, default='', help='The path of the output csv filename in generation data step')
    
    parser.add_argument('--output_exe_filename', '-oef', type=str, default='', help='The output csv filename in executing data step')
    parser.add_argument('--output_exe_dirname', '-oed', type=str, default='time', help='The dirname of the output csv filename in executing data step')
    parser.add_argument('--output_exe_path', '-oep', type=str, default='', help='The path of the output csv filename in executing data step')
    
    parser.add_argument('--iter_warmup', type=int, default=5, help='Number of iterations for warm-up')
    parser.add_argument('--iter_benchmark', '-b', type=int, default=10, help='Number of iterations for benchmark')
    parser.add_argument('--cpu', '-cpu', action="store_true", default=False, help='Force to use CPU to computate')
    # Profile Parameters
    parser.add_argument('--output_timeline_profile_dirname', '-otpd', type=str, default='timeline_profile', help='The dirname of timeline path')
    parser.add_argument('--output_timeline_profile_path', '-otpp', type=str, default='', help='The timeline path')
    parser.add_argument('--sleep_time', '-st', type=float, default=0.005, help='The sleep time for each profile data')
    # parser.add_argument('--log_level', '-ll', default='3', type=str, choices=['0', '1', '2', '3'], help='log level of tensorflow')
    # parser.add_argument('--backup_path', type=str, default=os.path.join(os.getcwd(), 'backup'), help='backup path')

    # Testing for memory copy
    parser.add_argument('--memory_copy_in', '-mci', type=int, default=1, help='the flags for copying data to gpu (0: for generate data in tf GPU)') 
    args = parser.parse_args()
    return args

def fullproof_flags(flags):
    default_dirname = 'data'
    flags.data_basename = flags.predition_layertype + '_' + flags.device
    if not flags.path:
        flags.path = os.path.join(os.getcwd(), default_dirname)

    if flags.gen_params:
        if not flags.output_params_filename:
            flags.output_params_filename = flags.predition_layertype + '_parameters.csv'
        if not flags.output_params_path:
            flags.output_params_path = os.path.join(flags.path, flags.output_params_dirname, flags.output_params_filename)

    if flags.exe_params:
        if not flags.input_params_filename:
            flags.input_params_filename = flags.predition_layertype + '_parameters.csv'
        if not flags.input_params_path:
            flags.input_params_path = os.path.join(flags.path, flags.input_params_dirname, flags.input_params_filename)
        if not flags.output_exe_filename:
            flags.output_exe_filename = flags.data_basename + '.csv'
        if not flags.output_exe_path:
            flags.output_exe_path = os.path.join(flags.path, flags.output_exe_dirname, flags.output_exe_filename)
    
    if flags.profile_params:
        if not flags.input_params_filename:
            flags.input_params_filename = flags.predition_layertype + '_parameters.csv'
        if not flags.input_params_path:
            flags.input_params_path = os.path.join(flags.path, flags.input_params_dirname, flags.input_params_filename)
        if not flags.output_timeline_profile_path:
            flags.output_timeline_profile_path = os.path.join(flags.path, flags.output_timeline_profile_dirname, flags.data_basename)
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
    print(success_tag + 'GPU is found') if tf.test.gpu_device_name() else print(warn_tag + 'GPU is Not found')
    return

def main():
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    success_tag = colored('[Success] ', 'green')
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

    if flags.exe_params or flags.profile_params:
        if not os.path.isfile(flags.input_params_path):
            ##prepare data from input csv
            print(warn_tag, "Please create random dataset or open the '-gp' tag for the supported layer at first!")
            return
        check_config(flags)
        df_data  = pd.read_csv(flags.input_params_path)
        params   = get_params(df_data.shape[0], flags.predition_layertype, is_train = flags.is_train)
        params.set_data(df_data)
        params.set_memcopyin(flags.memory_copy_in)
        params.set_colnames(get_colnames(flags.predition_layertype, is_train = flags.is_train))
        params.auto_generate_elements()
        params.generate_hashkey()
        #df_data2 = params.data
        #print(df_data2)
        #df_data2.to_csv("QQ.csv", index = None)
        #exit()

    if flags.exe_params:
        print("[Execution]")
        if os.path.isfile(flags.output_exe_path):
            print(warn_tag, "File: {} is Existed, Pleas delete it manually!".format((flags.output_exe_path)))
        else:
            #for index in range(params.data.shape[0]):
            for index in range(flags.num): # use flags.number as data size
                if index >= params.data.shape[0]:
                    print(warn_tag, 'over the dataframe size!')
                    break
                print('========== ', params.typename, index+1, '==========')
                tf.reset_default_graph()
                list_time = []
                if params.memcopyin:
                    input_ = params.get_input(index)
                op = params.get_tensor_from_index(index)
                sess = tf.Session()
                # session init 
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                try:
                    if params.memcopyin:
                        # Do WarmUp               
                        for _ in range(flags.iter_warmup):
                            sess.run(op, feed_dict = {params.inputs: input_})
                        # Do Benchmark
                        for _ in range(flags.iter_benchmark):
                            start_time = time.time()
                            sess.run(op, feed_dict = {params.inputs: input_})
                            list_time.append(((time.time()-start_time) * 1000)) # The unit of measurement is 'Millisecond'
                    else:
                        for _ in range(flags.iter_warmup):
                            sess.run(op)
                        # Do Benchmark
                        for _ in range(flags.iter_benchmark):
                            start_time = time.time()
                            sess.run(op)
                            list_time.append(((time.time()-start_time) * 1000)) # The unit of measurement is 'Millisecond'
                except tf.errors.ResourceExhaustedError:
                    continue
                list_time = np.array(list_time)
                time_data_ele = {
                    'hashkey':        str(params.data.loc[index, params.hash_colnames[0]]),
                    'time_max':       np.amax(list_time),
                    'time_min':       np.amin(list_time),
                    'time_median':    np.median(list_time),
                    'time_mean':      np.mean(list_time), 
                    'time_trim_mean': stats.trim_mean(list_time, 0.1),
                }
                df_ele = pd.DataFrame(data = time_data_ele, index=[0])
                print("time_mean: {} ms".format(time_data_ele['time_mean']))
                print(list_time)
                if index==0: 
                    df_ele.to_csv(flags.output_exe_path, index=None)
                else:
                    df_ele.to_csv(flags.output_exe_path, index=False, mode='a', header=False)
        print(success_tag, 'Execution is Done!')

    if flags.profile_params:
        print("[Profiling]")
        tf.reset_default_graph()
        run_metadata = tf.RunMetadata()
        #for index in range(params.data.shape[0]):
        for index in range(flags.num): # use flags.number as data size
            if index >= params.data.shape[0]:
                print(warn_tag, 'over the dataframe size!')
                break
            print('========== ', params.typename, index+1, '==========')
            tf.reset_default_graph()
            filename = os.path.join(flags.output_timeline_profile_path, str(params.data.loc[index, params.hash_colnames[0]]) + '.json')
            if os.path.isfile(filename):
                print(warn_tag, "File: {} is Existed, Pleas delete all duplicate files manually!".format((filename)))
                break

            if params.memcopyin:
                input_ = params.get_input(index)
            op = params.get_tensor_from_index(index)
            sess = tf.Session()
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            
            if params.memcopyin:
                # Do WarmUp
                for _ in range(flags.iter_warmup):
                    sess.run(op, feed_dict = {params.inputs: input_}) ## 注意: 這裡沒有option 跟run_metadata, 實際上沒差, 但邏輯上怪怪的 TBD
                # Do Benchmark            
                start_time = time.time()
                sess.run(op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata, feed_dict = { params.inputs: input_})
                print(((time.time()-start_time) * 1000), "ms") # The unit of measurement is 'Millisecond'

            else:
                # Do WarmUp
                for _ in range(flags.iter_warmup):
                    sess.run(op)## 注意: 這裡沒有option 跟run_metadata, 實際上沒差, 但邏輯上怪怪的 TBD
                # Do Benchmark            
                sess.run(op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    run_metadata=run_metadata)

            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(filename, 'w') as f:
                f.write(ctf) 
            time.sleep(flags.sleep_time)
        print(success_tag, 'Profiling is Done!')

if __name__ == '__main__':
    main()
