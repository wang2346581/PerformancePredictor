import os
import sys
import argparse
import numpy as np
import pandas as pd 
from termcolor import colored
from utils.utils import get_support_layers, get_colnames_from_dict
from utils.timeline_struct import Recorders

def read_preprocess_data_flags():
    parser = argparse.ArgumentParser('Preprocess Data Paremeters Parser')

    # General Parameters
    parser.add_argument('--predition_layertype', '-pl', default='convolution', type=str, choices=get_support_layers(), help='Layer types of neural networks')
    parser.add_argument('--device', '-d', type=str, default='1080ti', help='Device name as appearing in logfile')
    parser.add_argument('--parse_timeline', '-pt', action="store_true", default=False, help='parse timeline files')
    parser.add_argument('--combine', '-c', action="store_true", default=False, help='combine parameters, execution time, timeline data')
    parser.add_argument('--split', '-sp', action="store_true", default=False, help='split data to train and test data')
    parser.add_argument('--cpu', '-cpu', action="store_true", default=False, help='predicted device is cpu')
    parser.add_argument('--path', '-path', type=str, default='', help='The main path of this program')

    # Timeline Parser File and Path Parameters
    parser.add_argument('--is_train', '-is_train', type=int, default=0, help='collect optimizer data')
    parser.add_argument('--input_timeline_profile_dirname', '-itpd', type=str, default='timeline_profile', help='The input dirname of the timeline path')
    parser.add_argument('--input_timeline_profile_path', '-itpp', type=str, default='', help='The input path of the timeline path')
    parser.add_argument('--output_timeline_profile_filename', '-otpf', type=str, default='', help='The output filename of the timeline path')
    parser.add_argument('--output_timeline_profile_dirname', '-otpd', type=str, default='timeline_profile_csv', help='The output dirname of the timeline path')
    parser.add_argument('--output_timeline_parser_path', '-otpp', type=str, default='', help='The output path of the timeline path')

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

    # Combine Parameters
    parser.add_argument('--combine_input_params_filename', '-cipf', type=str, default='', help='[Combination] The input csv file name')
    parser.add_argument('--combine_input_params_dirname', '-cipd', type=str, default='struct', help='[Combination] The dirname of the input csv filename in generation data step')
    parser.add_argument('--combine_input_params_path', '-cipp', type=str, default='', help='[Combination] The path of the input csv filename in generation data step')
    parser.add_argument('--combine_input_exe_filename', '-cief', type=str, default='', help='[Combination] The input csv filename in executing data step')
    parser.add_argument('--combine_input_exe_dirname', '-cied', type=str, default='time', help='[Combination] The dirname of the input csv filename in executing data step')
    parser.add_argument('--combine_input_exe_path', '-ciep', type=str, default='', help='[Combination] The path of the input csv filename in executing data step')
    parser.add_argument('--combine_input_timeline_profile_filename', '-citpf', type=str, default='', help='[Combination] The input filename of the timeline path')
    parser.add_argument('--combine_input_timeline_profile_dirname', '-citpd', type=str, default='timeline_profile_csv', help='[Combination] The input dirname of the timeline path')
    parser.add_argument('--combine_input_timeline_profile_path', '-citpp', type=str, default='', help='[Combination] The input path')

    parser.add_argument('--combine_output_filename', '-cof', type=str, default='all.csv', help='[Combination] The filename of the output path')
    parser.add_argument('--combine_output_dirname', '-cod', type=str, default='', help='[Combination] The dirname of the output path')
    parser.add_argument('--combine_output_path', '-cop', type=str, default='', help='[Combination] The output path')
    
    # Split Parameters
    parser.add_argument('--split_proportion', '-spp', type=float, default=0.2, help='[Split] The data proportion of the test data')
    parser.add_argument('--split_input_filename', '-sif', type=str, default='all.csv', help='[Split] The input combined csv filename')
    parser.add_argument('--split_input_dirname', '-sid', type=str, default='', help='[Split] The input dirname of the combined csv filename')
    parser.add_argument('--split_input_path', '-sip', type=str, default='', help='[Split] The input path')
    parser.add_argument('--split_output_train_filename', '-sotrf', type=str, default='train.csv', help='[Split] The output train csv filename')
    parser.add_argument('--split_output_test_filename', '-sottf', type=str, default='test.csv', help='[Split] The output test csv filename')
    parser.add_argument('--split_output_dirname', '-sod', type=str, default='', help='[Split] The output dirname of the combined csv filename')
    parser.add_argument('--split_output_train_path', '-sotrp', type=str, default='', help='[Split] The output train path')
    parser.add_argument('--split_output_test_path', '-sottp', type=str, default='', help='[Split] The output test path')

    args = parser.parse_args()
    return args

def fullproof_flags(flags):
    default_dirname = 'data'
    flags.data_basename = flags.predition_layertype + '_' + flags.device
    if not flags.path:
        flags.path = os.path.join(os.getcwd(), default_dirname)

    if flags.parse_timeline:
        if not flags.input_timeline_profile_path:
            flags.input_timeline_profile_path = os.path.join(flags.path, flags.input_timeline_profile_dirname, flags.data_basename)
        if not flags.output_timeline_profile_filename:
            flags.output_timeline_profile_filename = flags.data_basename + '.csv'
        if not flags.output_timeline_parser_path:
            flags.output_timeline_parser_path = os.path.join(flags.path, flags.output_timeline_profile_dirname, flags.data_basename, flags.output_timeline_profile_filename)

    if flags.combine:
        if not flags.combine_input_params_filename:
            flags.combine_input_params_filename = flags.predition_layertype + '_parameters.csv'
        if not flags.combine_input_params_path:
            flags.combine_input_params_path = os.path.join(flags.path, flags.combine_input_params_dirname, flags.combine_input_params_filename)
        if not flags.combine_input_exe_filename:
            flags.combine_input_exe_filename = flags.data_basename + '.csv'
        if not flags.combine_input_exe_path:
            flags.combine_input_exe_path = os.path.join(flags.path, flags.combine_input_exe_dirname, flags.combine_input_exe_filename)
        if not flags.combine_input_timeline_profile_filename:
            flags.combine_input_timeline_profile_filename = flags.data_basename + '.csv'
        if not flags.combine_input_timeline_profile_path:
            flags.combine_input_timeline_profile_path = os.path.join(flags.path, flags.combine_input_timeline_profile_dirname, flags.data_basename, flags.combine_input_timeline_profile_filename)
    
        if not flags.combine_output_dirname:
            flags.combine_output_dirname = flags.data_basename
        if not flags.combine_output_path:
            flags.combine_output_path = os.path.join(flags.path, flags.combine_output_dirname, flags.combine_output_filename)

    if flags.split:
        if not flags.split_input_dirname:
            flags.split_input_dirname = flags.data_basename
        if not flags.split_input_path:
            flags.split_input_path = os.path.join(flags.path, flags.split_input_dirname, flags.split_input_filename)
        if not flags.split_output_dirname:
            flags.split_output_dirname = flags.data_basename
        if not flags.split_output_train_path:
            flags.split_output_train_path = os.path.join(flags.path, flags.split_output_dirname, flags.split_output_train_filename)
        if not flags.split_output_test_path:
            flags.split_output_test_path = os.path.join(flags.path, flags.split_output_dirname, flags.split_output_test_filename)

    return flags

def auto_create_dir(flags):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)
    if flags.parse_timeline:
        create_dir_elemenet(os.path.dirname(flags.output_timeline_parser_path))
    if flags.combine:
        create_dir_elemenet(os.path.dirname(flags.combine_output_path))
    if flags.split:
        create_dir_elemenet(os.path.dirname(flags.split_output_train_path))

def main():
    warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
    success_tag = colored('[Success] ', 'green')
    flags = read_preprocess_data_flags()
    flags = fullproof_flags(flags)
    print(flags)
    auto_create_dir(flags)
    if flags.parse_timeline:
        print('[Parse Timeline]')
        files_input_timeline_profile_path = os.listdir(flags.input_timeline_profile_path)
        # print(all_files)
        index = 0
        for filename in files_input_timeline_profile_path:
            full_filename = os.path.join(flags.input_timeline_profile_path, filename)
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
            
            if recorders.fail_data == True: ### Pass the  Fail data 
                print(recorders)
                print("fail data index is {}".format(index))
                continue

            for key, value in recorders.data.items():
                if key == 'hashkey':
                    continue
                recorders.data[key] = value / 1000 # The unit of measurement is 'Millisecond' 
            df_ele = pd.DataFrame(data = recorders.data, index=[0])
            if index==0: 
                df_ele.to_csv(flags.output_timeline_parser_path, index=False)
                index += 1
            else:
                df_ele.to_csv(flags.output_timeline_parser_path, index=False, mode='a', header=False)
        print(success_tag, 'Parse Timeline is Done!')
    
    if flags.combine:
        print('[Combine all data]')
        if os.path.isfile(flags.combine_output_path):
            print(warn_tag, "File: {} is Existed, Pleas delete it manually!".format((flags.combine_output_path)))
        else:
            col_dict  = get_colnames_from_dict(is_train = flags.is_train)
            df_struct = pd.read_csv(flags.combine_input_params_path)
            df_all = df_struct.copy()
            if os.path.isfile(flags.combine_input_exe_path): ### TBD 流程, 還可以更好
                df_time     = pd.read_csv(flags.combine_input_exe_path)
                for ct in col_dict['time']:
                    df_all[ct] = df_all.hashkey.map(df_time.set_index('hashkey')[ct])
            else:
                print(warn_tag, "No time data is also acceptable for convience!")
            if not flags.cpu: ## 不是cpu才有timeline可以合併
                df_timeline = pd.read_csv(flags.combine_input_timeline_profile_path)
                for ct in col_dict['profile']:
                    df_all[ct] = df_all.hashkey.map(df_timeline.set_index('hashkey')[ct])
            df_all.to_csv(flags.combine_output_path, index=False)
            print(success_tag, 'Combination is Done!')
    
    if flags.split:
        if os.path.isfile(flags.split_output_test_path):
            print(warn_tag, "File: {} is Existed, Pleas delete it manually!".format((flags.split_output_test_path)))
        elif os.path.isfile(flags.split_output_train_path):
            print(warn_tag, "File: {} is Existed, Pleas delete it manually!".format((flags.split_output_train_path)))
        else:
            print('[Split data as train and test data]')
            df = pd.read_csv(flags.split_input_path)
            df = df.dropna()
            df = df.reset_index(drop=True)
            df_test  = df.loc[0:int(flags.split_proportion * len(df))-1, :]
            df_train = df.loc[int(flags.split_proportion * len(df)):len(df)-1, :]
            df_test.to_csv(flags.split_output_test_path, index = None)
            df_train.to_csv(flags.split_output_train_path, index = None)
            print(success_tag, 'Split data is Done!')
    
if __name__ == '__main__':
    main()