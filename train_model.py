import os
import sys
import shutil
import logging
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from termcolor import colored


from scipy import stats
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from utils.model import Model
from utils.meter import RegErrorMeter, ExpRegErrorMeter, DExpRegErrorMeter
from utils.utils import get_feature_target, _logger, _set_file, get_support_accuracy_metrics
from utils.network import get_nn_list
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_train_model_flags():   
    parser = argparse.ArgumentParser(description='Deep learning for predicting the performance of neural network')
    #Basic Parameters
    parser.add_argument('--predition_device', '-pd', default='1080ti', 
                    type=str, help='predition device for training or testing')
    parser.add_argument('--network_name', '-n', default='perfnetA',
                    type=str, choices=get_nn_list(), help='network name for training or testing')
    parser.add_argument('--loss_name', '-lf', default='maple',
                    type=str, choices=['msle', 'mse', 'maple', 'poisson', 'mae'], help='loss function name for training or testing')
 
    parser.add_argument('--predition_layertype', '-pl', default='convolution', 
                    type=str, choices=['convolution', 'dense', 'pooling'],
                    help='predition layer type for training or testing')

    parser.add_argument('--predition_sub_layertype', '-psl', default='', 
                    type=str, choices=['pre', 'exe', 'post', 'back', 'for', ''],
                    help='predition layer type for training or testing')

    parser.add_argument('--start_epoch', default=0, 
                        type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', '-e', default=200, 
                        type=int,  help='number of total epochs to run')
    parser.add_argument('--batch_size', '-b', default=128, 
                        type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--use_target', '-ut', default='time', 
                        type=str, help='target value for prediction')
    
    
    #Data path Parameters
    parser.add_argument('--data_dirname', '-dd', default='data', 
        type=str, help='data dirname')
    parser.add_argument('--train_csv', '-trcsv', default='train.csv', type=str,
                        help='train csv filename')
    parser.add_argument('--test_csv', '-tecsv', default='test.csv', type=str,
                        help='test csv filename')
    parser.add_argument('--train_csv_dirname', '-trcd', type=str, default='', help='The train data dirname of the combined csv filename')
    parser.add_argument('--test_csv_dirname', '-tecd', type=str, default='', help='The test data dirname of the combined csv filename')
    parser.add_argument('--train_csv_path', '-trcp', type=str, default='', help='The output train path')
    parser.add_argument('--test_csv_path', '-tecp', type=str, default='', help='The output test path')


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



    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='Full file name to latest checkpoint (default: none)')
    parser.add_argument('--save_model', '-s', default=1, 
                        type=int, help='save model flags')
    parser.add_argument('--load_model', '-l', action='store_true', 
                        help='load from checkpoint')

    parser.add_argument('--backup_model', '-bm', default=1, 
                        type=int, help='backup the  checkpoint')

    parser.add_argument('--backup_model_dirname', '-bmd', default='backup_model', 
        type=str, help='the backup model dirname')

    parser.add_argument('--backup_model_path', '-bmp', default='', 
                        type=str, help='the backup model path')

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

    #Learning rate Parameters
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

    #Inference or Training 
    parser.add_argument('-eva', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    # Log file Parameters
    parser.add_argument('--log2file', '-log2file', default=1, 
                        type=int, help='log the whole events')
    parser.add_argument('--log_filename', '-log_filename', default='', 
                        type=str, help='the log filename')
    parser.add_argument('--log_dirname', '-log_dirname', default='log_file', 
                        type=str, help='the log dirname')
    parser.add_argument('--log_path', '-log_path', default='', 
                        type=str, help='the log path')            

    # Predict which model
    parser.add_argument('--model', type=str, default='', help='Device name as appearing in logfile')

    flags = parser.parse_args()
    return flags

def fullproof_flags(flags):
    ### foolproof for simple flags
    if flags.evaluate:
        flags.save_model   = 0
        flags.load_model   = 1
    if not flags.save_model:
        flags.backup_model = 0

    ### data path
    #default_data_dirname = 'data'
    flags.data_basename = flags.predition_layertype + '_' + flags.predition_device
    flags.network_loss_name = flags.network_name + '_' + flags.loss_name

    if flags.predition_sub_layertype:
        flags.data_ex_basename = flags.predition_layertype + '_' + flags.predition_sub_layertype + '_' + flags.predition_device 
    else:
        flags.data_ex_basename = flags.data_basename
    
    if not flags.train_csv_path:
        if not flags.train_csv_dirname:
            flags.train_csv_dirname = flags.data_basename
        flags.train_csv_path = os.path.join(os.getcwd(), flags.data_dirname, flags.train_csv_dirname, flags.train_csv)

    if not flags.test_csv_path:
        if not flags.test_csv_dirname:
            flags.test_csv_dirname = flags.data_basename
        flags.test_csv_path = os.path.join(os.getcwd(), flags.data_dirname, flags.test_csv_dirname, flags.test_csv)
    
    ### Model Path
    if not flags.model_path:
        flags.model_path = os.path.join(os.getcwd(), flags.model_dirname, flags.data_ex_basename)

    if not flags.backup_model_path:
        flags.backup_model_path = os.path.join(os.getcwd(), flags.backup_model_dirname, flags.data_ex_basename)
    
    flags.ckpt_filename = 'checkpoint'
    flags.mae_dirname   = flags.network_loss_name + '_mae'
    flags.mape_dirname  = flags.network_loss_name + '_mape'
    flags.rmse_dirname  = flags.network_loss_name + '_rmse'
    flags.r2_dirname    = flags.network_loss_name + '_r2'
    flags.mae_path  = os.path.join(flags.model_path, flags.mae_dirname, flags.ckpt_filename)
    flags.mape_path = os.path.join(flags.model_path, flags.mape_dirname, flags.ckpt_filename)
    flags.rmse_path = os.path.join(flags.model_path, flags.rmse_dirname, flags.ckpt_filename)
    flags.r2_path   = os.path.join(flags.model_path, flags.r2_dirname, flags.ckpt_filename)
    
    if flags.load_model:
        if not flags.resume:
            flags.resume = os.path.join(flags.model_path, flags.network_loss_name + '_' + flags.accuracy_metric)

    ### Feature and Target json
    if not flags.ft_filepath:
        if not flags.ft_filename:
            flags.ft_filename = flags.predition_layertype
        if flags.predition_sub_layertype:
            flags.ft_filename = flags.ft_filename + '_' + flags.predition_sub_layertype
        flags.ft_filename = flags.ft_filename + '.json'
        flags.ft_filepath = os.path.join(os.getcwd(), flags.ft_dirname, flags.ft_filename)

    ### Log File Path 
    if flags.log2file:
        if not flags.log_filename:
            if flags.evaluate:
                flags.log_filename = flags.data_ex_basename + '_' + flags.network_name + '_' + flags.loss_name + '_testing.log'
            else:
                flags.log_filename = flags.data_ex_basename + '_' + flags.network_name + '_' + flags.loss_name + '.log'
        if not flags.log_path:
            flags.log_path = os.path.join(os.getcwd(), flags.log_dirname, flags.log_filename)
        
    return flags

def auto_create_dir(flags):
    def create_dir_elemenet(path):
        warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
        if not os.path.isdir(path):
            os.makedirs(path)
            print(warn_tag + 'Auto create dir: ' + path)

    if flags.log2file:
        create_dir_elemenet(os.path.dirname(flags.log_path))
        _logger.info("==> Creating the log file...")
        _set_file(flags.log_path)

    if flags.save_model:
        if flags.backup_model and not flags.load_model:
            create_dir_elemenet(flags.backup_model_path)
            if os.path.isdir(os.path.dirname(flags.mae_path)):
               shutil.copytree(os.path.dirname(flags.mae_path), os.path.join(flags.backup_model_path, flags.mae_dirname + '.' + datetime.now().strftime('%m%d-%H%M%S')))
            if os.path.isdir(os.path.dirname(flags.mape_path)):
                shutil.copytree(os.path.dirname(flags.mape_path), os.path.join(flags.backup_model_path, flags.mape_dirname + '.' + datetime.now().strftime('%m%d-%H%M%S')))
            if os.path.isdir(os.path.dirname(flags.rmse_path)):
                shutil.copytree(os.path.dirname(flags.rmse_path), os.path.join(flags.backup_model_path, flags.rmse_dirname + '.' + datetime.now().strftime('%m%d-%H%M%S')))
            if os.path.isdir(os.path.dirname(flags.r2_path)):
                shutil.copytree(os.path.dirname(flags.r2_path), os.path.join(flags.backup_model_path, flags.r2_dirname + '.' + datetime.now().strftime('%m%d-%H%M%S')))
            _logger.info("[BackUp] Existing model '{}' backuped to '{}'".
                    format(flags.model_path, flags.backup_model_path))
        create_dir_elemenet(os.path.dirname(flags.mae_path))
        create_dir_elemenet(os.path.dirname(flags.mape_path))
        create_dir_elemenet(os.path.dirname(flags.rmse_path))
        create_dir_elemenet(os.path.dirname(flags.r2_path))

def error_reset(flags):
    flags.start_epoch = 0
    flags.best_r2   = -sys.maxsize - 1
    flags.best_mape = sys.maxsize
    flags.best_mae  = sys.maxsize
    flags.best_rmse = sys.maxsize
    return flags

def store_sess(saver, sess, path, global_step, save_model = 1, save_str='r2'):
    if save_model:
        _logger.info('[Save] Best:{}, path:{}'.format(save_str, path))
        saver.save(sess, path, global_step = global_step)

def data_preparation(flags, feature=list(), target=str()):
    _logger.info("==> Do data preparation ...")
    ### Get Data from csv
    df_train = pd.read_csv(flags.train_csv_path)
    df_test  = pd.read_csv(flags.test_csv_path)

    if not feature or not target: ### For Fool Proofing
        feature, target = get_feature_target(os.path.join(os.getcwd(), "Feature_Target", "conv.json"))
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", feature, target)

    ### Feature transformations
    ft_str = "[Features] Transformation: {} to Features: {}".format(flags.feature_transform, flags.ft_list)
    if not flags.feature_transform:
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
    _logger.info(ft_str)

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
    _logger.info(fp_str)

    ### Feature StandardScale
    _logger.info("[Features] Standardscale to all Features")
    scaler = StandardScaler()
    scaler.fit(train_f.astype(float))

    train_scale = scaler.transform(train_f.astype(float))
    test_scale  = scaler.transform(test_f.astype(float))

    feature_size = train_f.shape[1]
    ### Target Transformation
    _logger.info("[Target] target time * {}".format(flags.magic_scaler)) 
    df_train[target] = df_train[target] * flags.magic_scaler ###for convenience
    df_test[target]  = df_test[target]  * flags.magic_scaler ###for convenience

    if flags.target_transform == 'log':
        _logger.info("[Target] logarithm the traget time ") 
        df_train[target] = np.log(df_train[target])
        df_test[target]  = np.log(df_test[target])
    return train_scale, df_train[target], test_scale, df_test[target], feature_size, df_train, df_test

def get_meter(lossfunc, transform, magic_scaler = 1):
    if lossfunc == 'poisson':
        return ExpRegErrorMeter(magic_scaler = magic_scaler)
    elif transform == 'log':
        return DExpRegErrorMeter(magic_scaler = magic_scaler)
    return RegErrorMeter(magic_scaler = magic_scaler)

def validation(flags, model, testdata, testlabel): 
    _logger.info("==> Do inference...")
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    te_num_datapoints = testdata.shape[0]
    te_list_datapoints = np.arange(0,te_num_datapoints)
    te_num_batches = np.int(np.ceil(te_num_datapoints/model.batch_size))
    with tf.Session() as sess:
        sess.run(init)
        if flags.load_model:
            _logger.info("==> Resuming model from checkpoint..")
            print(flags.resume)
            saver.restore(sess, tf.train.latest_checkpoint(flags.resume))
        regm_test = get_meter(flags.loss_name, flags.target_transform, flags.magic_scaler) #RegErrorMeter()
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
        _logger.info('[_Val_] Loss: {:.3f} | Mae: {:.3f} | Mape : {:.3f}% | RMSe: {:.3f} | R2: {:.3f}' \
            .format(regm_test.loss.avg, regm_test.error.mae, regm_test.error.mape, regm_test.error.rmse, regm_test.error.r2))
    return regm_test.error.prediction, regm_test.error.answer

def train(flags, model, traindata, trainlabel, testdata, testlabel):
    _logger.info("==> Train the data...")
    model.train_op
    init = tf.global_variables_initializer()

    #save model 
    if flags.save_model:
        saver_mae  = tf.train.Saver(max_to_keep=1)
        saver_mape  = tf.train.Saver(max_to_keep=1)
        saver_rmse = tf.train.Saver(max_to_keep=1)
        saver_r2   = tf.train.Saver(max_to_keep=1)
    else:
        saver_mae = saver_mape = saver_rmse = saver_r2 = None

    # For training
    tr_num_datapoints = traindata.shape[0]
    list_datapoints = np.arange(0,tr_num_datapoints)
    tr_num_batches = np.int(np.ceil(tr_num_datapoints/model.batch_size))

    #For Validation
    te_num_datapoints = testdata.shape[0]
    te_list_datapoints = np.arange(0,te_num_datapoints)
    te_num_batches = np.int(np.ceil(te_num_datapoints/model.batch_size))

    # Training and Validation 
    with tf.Session() as sess:
        sess.run(init)
        
        if flags.load_model:
            saver = tf.train.Saver()
            _logger.info("==> Resuming model from checkpoint..")
            saver.restore(sess, tf.train.latest_checkpoint(flags.resume))
            ckpt = tf.train.get_checkpoint_state(flags.resume)
           
            regm_test = get_meter(flags.loss_name, flags.target_transform)
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
            flags.start_epoch = int(os.path.basename(ckpt.all_model_checkpoint_paths[-1]).split('-')[1])
            flags.best_r2   = regm_test.error.r2
            flags.best_mape   = regm_test.error.mape
            flags.best_mae = regm_test.error.mae
            flags.best_rmse = regm_test.error.rmse
            _logger.info('[Params] all parameters are updated to {}'.format(flags))
        

        for epoch in range(flags.start_epoch, flags.epochs):
            _logger.info('\nEpoch:{}'.format(epoch + 1))
            np.random.shuffle(list_datapoints)
            regm = get_meter(flags.loss_name, flags.target_transform, flags.magic_scaler)
            #adjust the learning rate
            model.adjust_lr(epoch)
            # train for one epoch
            for i in range(0, tr_num_batches):
                cur_batch_size = min((i+1)*model.batch_size,tr_num_datapoints) - (i*model.batch_size)
                batch = list_datapoints[i*model.batch_size:min((i+1)*model.batch_size,tr_num_datapoints)]
                _, trainloss, train_pred_time = sess.run(
                        [model.train_op, model.loss, model.prediction],
                            feed_dict={
                                model.tf_inputs: traindata[batch,:],
                                model.tf_targets: trainlabel[batch],
                                model.tf_lr: model.learning_rate, 
                                model.tf_istraining: True})
                
                # update to meter
                regm.error.update(trainlabel[batch], train_pred_time)
                regm.loss.update(trainloss, cur_batch_size)

            regm.error.summary()
            _logger.info('[Train] Loss: {:.3f} | Mae: {:.3f} | Mape : {:.3f}% | RMSe: {:.3f} | R2: {:.3f}' \
                .format(regm.loss.avg, regm.error.mae, regm.error.mape, regm.error.rmse, regm.error.r2))

            # Validate it
            regm_test = get_meter(flags.loss_name, flags.target_transform, flags.magic_scaler)
            for i in range(0, te_num_batches):
                cur_batch_size = min((i+1)*model.batch_size,te_num_datapoints) - (i*model.batch_size)
                batch = te_list_datapoints[i*model.batch_size:min((i+1)*model.batch_size,te_num_datapoints)]
                testloss, test_pred_time = sess.run(
                    [model.loss, model.prediction],
                        feed_dict={
                            model.tf_inputs: testdata[batch,:],
                            model.tf_targets: testlabel[batch],
                            model.tf_istraining: False})
                # print('test_pred_time', test_pred_time)
                # update to meter
                regm_test.error.update(testlabel[batch], test_pred_time)
                regm_test.loss.update(testloss, cur_batch_size)
            regm_test.error.summary()
            
            _logger.info('[_Val_] Loss: {:.3f} | Mae: {:.3f} | Mape : {:.3f}% | RMSe: {:.3f} | R2: {:.3f}' \
                .format(regm_test.loss.avg, regm_test.error.mae, regm_test.error.mape, regm_test.error.rmse, regm_test.error.r2))
            
          
            if flags.best_mae >= regm_test.error.mae:
                flags.best_mae = regm_test.error.mae
                store_sess(saver_mae, sess, flags.mae_path, epoch+1, save_model = flags.save_model, save_str='mae')
            if flags.best_mape >= regm_test.error.mape:
                flags.best_mape = regm_test.error.mape
                store_sess(saver_mape, sess, flags.mape_path, epoch+1, save_model = flags.save_model, save_str='mape')
            if flags.best_rmse >= regm_test.error.rmse:
                flags.best_rmse = regm_test.error.rmse
                store_sess(saver_rmse, sess, flags.rmse_path, epoch+1, save_model = flags.save_model, save_str='rmse')
            if flags.best_r2 <= regm_test.error.r2:
                flags.best_r2 = regm_test.error.r2
                store_sess(saver_r2, sess, flags.r2_path, epoch+1, save_model = flags.save_model, save_str='r2')

    _logger.info('[Summary_Best] Mae: {:.3f} | Mape : {:.3f}% | RMSe: {:.3f} | R2: {:.3f}' \
        .format(flags.best_mae, flags.best_mape, flags.best_rmse, flags.best_r2))

def main():
    flags = read_train_model_flags()
    flags = fullproof_flags(flags)
    ### Basic Setting 
    auto_create_dir(flags)
    #model_config(flags) # model path config
    flags = error_reset(flags)  # error meter reset
    feature, target = get_feature_target(flags.ft_filepath) 

    ###  Data Prepare
    train_feature, train_target, test_feature, test_target, feature_size, df_train, df_test = data_preparation(flags, feature, target)
         
    ### Get Model
    model = Model(flags, feature_size)
    _logger.info('[Params] Init parameters are {}'.format(flags))
    
    ### Evalidation 
    if flags.evaluate:
        pred_time, answer_time = validation(flags, model, test_feature, test_target)

        df_test['pred_time'] = pred_time
        #df_test['ans_time'] = answer_time

        #pred_time, answer_time = validation(flags, model, train_feature, train_target)
        #df_train['pred_time'] = pred_time

        # pred_time, answer_time = validation(flags, model, train_feature, train_target)
        # df_train['pred_time'] = pred_time
        # df_train['ans_time'] = answer_time
        #print(df_test)
        #df_train.to_csv('./train.csv', index = None)
        df_test.to_csv('./QQQQQtest.csv', index = None)
        return 
    else:
        #print(test_feature, test_target)
        train(flags, model, train_feature, train_target, test_feature, test_target)

if __name__ == '__main__':
    main()
