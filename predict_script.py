import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
import os
import argparse
from utils.utils import get_support_batchsize, get_verify_models
from utils.network import get_nn_list

parser = argparse.ArgumentParser('Model Verify Parameters Parser')
# General parameters
parser.add_argument('--device', '-d', type=str, default='1080ti', help='target device Name for benchmarking')
parser.add_argument('--network_name', '-n', default='perfnetA', type=str, choices=get_nn_list(), help='network name for training or testing')
parser.add_argument('--loss_name', '-lf', default='malpe', type=str, choices=['msle', 'mse', 'malpe', 'poisson', 'mae'], help='loss function name for training or testing')
args = parser.parse_args()
batch_list = get_support_batchsize()
net_list = get_verify_models()
net_list = ['vgg16']

for net in net_list:
	for batch in batch_list:
	
		os.system('python3 verify_model.py -pdm -n %s -d %s -l -eva -lf %s --model %s -b %d' \
			% (args.network_name, args.device, args.loss_name, net, batch))
