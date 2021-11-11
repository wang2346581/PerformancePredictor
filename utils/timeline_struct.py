import os 
import re
import sys
import copy
import json

class RecordersTesting(object):
    """ "Store Data infos """
    def __init__(self, json_filename = None, json_data = None,
                str_replica_cpu = '(replica:0)*(CPU:0)+ (Compute)+', 
                str_all_compute    = '(GPU:0)*(all Compute)',
                str_memcpy         = '(memcpy)+ (Compute)+',
                str_compute_transpose_in   = 'TransposeNHWCToNCHW',
                str_compute_transpose_out  = 'TransposeNCHWToNHWC',
                str_memcpyH2D      = 'MEMCPYHtoD',
                str_memcpyD2H      = 'MEMCPYDtoH',
                str_retval         = 'retval'):
        self.data      = dict()
        self.fail_data = False
        self.eztags = EasyTagsTesting(json_filename, json_data, str_replica_cpu, str_all_compute, str_memcpy,
                    str_compute_transpose_in, str_compute_transpose_out,
                    str_memcpyH2D, str_memcpyD2H, str_retval)
        self.init_time = self.eztags.init_time
        ### Search First and Last GPU Step
        self.first_gpu_step = EasyStepTag('first_gpu', None, self.init_time, start_time = sys.maxsize)
        self.last_gpu_step  = EasyStepTag('last_gpu',  None, self.init_time, start_time = 0)

        self.inputs_pattern   = 'inputs'
        self.outputs_pattern  = 'edge_'
        
        if self.eztags.all_compute.existed:
            for item in self.eztags.all_compute.components:
                if item.__contains__('ts') and item.__contains__('dur'):
                    time_now = item['ts'] - self.init_time
                    if time_now < self.first_gpu_step.start_time:
                        if item['args'].__contains__('name') and re.search(self.inputs_pattern, item['args']['name'], re.M | re.I): ### TBD
                                self.eztags.memcpyH2D.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                                continue
                        if self.eztags.compute_transpose_in.existed:
                            if not re.search(self.eztags.str_compute_transpose_in, item['args']['name'], re.M|re.I):
                               self.first_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                        else:
                            self.first_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                    
                    if time_now >= self.last_gpu_step.start_time:
                        if item['args'].__contains__('name') and re.search(self.outputs_pattern, item['args']['name'], re.M | re.I):### TBD
                                self.eztags.memcpyD2H.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                                continue
                        self.last_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
        self.calculate_result()
        
    def calculate_result(self):
        self.preprocess_time = self.first_gpu_step.start_time
        if self.eztags.compute_transpose_out.existed and not self.eztags.compute_transpose_out.has_calculated:
            self.eztags.compute_transpose_out.has_calculated = True
            self.tranposeout_time = self.eztags.compute_transpose_out.end_time - self.eztags.compute_transpose_out.last_end
        else:
            self.tranposeout_time = 0
        self.execution_time = self.last_gpu_step.end_time - self.first_gpu_step.start_time - self.tranposeout_time 
        if self.eztags.memcpyD2H.existed:
            self.memcpyD2H_time  = self.eztags.memcpyD2H.end_time - self.last_gpu_step.end_time
            self.retval_time     = self.eztags.retval.end_time - self.eztags.memcpyD2H.end_time
        else:
            self.memcpyD2H_time = 0
            self.retval_time    = self.eztags.retval.end_time - self.last_gpu_step.end_time
        self.retval_half_time = self.retval_time / 2 

        min_post_start_time = min(self.eztags.retval.start_time, self.eztags.memcpyD2H.start_time)
        if min_post_start_time < self.last_gpu_step.end_time:
            self.fail_data = True
        self.sess_time = max(self.eztags.retval.end_time, self.eztags.memcpyD2H.end_time, self.last_gpu_step.end_time) ### TBD

        self.data = {
            'hashkey':            os.path.splitext(os.path.basename(self.eztags.json_filename))[0],
            'preprocess_time':    self.preprocess_time,
            'execution_time':     self.execution_time,
            'memcpy_time':        self.memcpyD2H_time, 
            'retval_time':        self.retval_time,
            'retval_half_time':   self.retval_half_time,
            'memcpy_retval':      self.memcpyD2H_time + self.retval_time,
            'memcpy_retval_half': self.memcpyD2H_time + self.retval_half_time,
            'sess_time':          self.sess_time
        }
       
    def __str__(self):
        tmp_str  = self.eztags.__str__()
        tmp_str +=  self.first_gpu_step.__str__()
        tmp_str +=  self.last_gpu_step.__str__()
        tmp_str += "*" * 40 + "\n" + "[Result]\n"
        tmp_str += "Fail Data:        {}\n".format(self.fail_data)
        tmp_str += "tranposeout_time: {} ms\n".format(self.tranposeout_time/1000)
        tmp_str += "preprocess:       {} ms\n".format(self.preprocess_time/1000)
        tmp_str += "execution_time:   {} ms\n".format(self.execution_time/1000)
        tmp_str += "memcpy_time:      {} ms\n".format(self.memcpyD2H_time/1000)
        tmp_str += "retval_time:      {} ms\n".format(self.retval_time/1000)
        tmp_str += "retval_half_time: {} ms\n".format(self.retval_half_time/1000)
        tmp_str += "session time      {} ms".format(self.sess_time/1000)
        return tmp_str

class EasyTagsTesting(object):
    """"Tags of all important process name"""
    def __init__(self, json_filename = None, json_data = None, 
                    str_replica_cpu = '(replica:0)*(CPU:0)+ (Compute)+', 
                    str_all_compute    = '(GPU:0)*(all Compute)',
                    str_memcpy         = '(memcpy)+ (Compute)+',
                    str_compute_transpose_in   = 'TransposeNHWCToNCHW',
                    str_compute_transpose_out  = 'TransposeNCHWToNHWC',
                    str_memcpyH2D      = 'MEMCPYHtoD',
                    str_memcpyD2H      = 'MEMCPYDtoH',
                    str_retval         = 'retval'):
        
        self.init_time  = sys.maxsize
        self.json_filename = json_filename
        self.json_data  = json_data
        self.str_compute_transpose_in   = str_compute_transpose_in
        self.str_compute_transpose_out  = str_compute_transpose_out
        self.str_memcpyH2D = str_memcpyH2D
        self.str_memcpyD2H = str_memcpyD2H
        self.str_retval    = str_retval
        self.replica_cpu   = EasyTagTitle('replica_cpu', str_replica_cpu)
        self.all_compute   = EasyTagTitle('all_compute', str_all_compute)
        self.memcpy        = EasyTagTitle('memcpy', str_memcpy)
        self.compute_transpose_in  = None
        self.compute_transpose_out = None
        self.memcpyH2D     = None
        self.memcpyD2H     = None
        self.retval        = None
        self.quick_reset()
        self.setup_steps()

    def init_json_data(self):
        if not self.json_data:
            if not self.json_filename:
                raise FileNotFoundError("Json Data Not Found!!")
            with open(self.json_filename, 'r') as f:
                self.json_data = json.load(f)

    def reset_init_time(self):
        for item in self.json_data['traceEvents']:
            # The smallest time as the start time
            if item.__contains__('ts') and item['ts'] < self.init_time:
                self.init_time = item['ts'] 
    
    def quick_reset(self):
        if not self.json_data:
            self.init_json_data()
        # Check all title and set init time 
        for item in self.json_data['traceEvents']:
            if item.__contains__('ts') and item['ts'] < self.init_time:
                self.init_time = item['ts'] 
            if item.__contains__('name') and item['name'] == 'process_name':
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                if re.search(self.all_compute.pattern, item['args']['name'], re.M|re.I):
                    self.all_compute.pid = item['pid']
                    self.all_compute.components = self.json_data
                if re.search(self.memcpy.pattern, item['args']['name'], re.M|re.I):
                    self.memcpy.pid = item['pid']
                    self.memcpy.components = self.json_data
                if re.search(self.replica_cpu.pattern, item['args']['name'], re.M|re.I):
                    self.replica_cpu.pid = item['pid']
                    self.replica_cpu.components = self.json_data

        self.compute_transpose_in  = EasyStepTag('compute_transpose_in',  self.str_compute_transpose_in, self.init_time)
        self.compute_transpose_out = EasyStepTagTrace('compute_transpose_out', self.str_compute_transpose_out, self.init_time)
        self.memcpyH2D     = EasyStepTag('memcpyH2D', self.str_memcpyH2D, self.init_time)
        self.memcpyD2H     = EasyStepTag('memcpyD2H', self.str_memcpyD2H, self.init_time)
        self.retval        = EasyStepTag('retval', self.str_retval, self.init_time)
        del self.json_data
    
    def setup_steps(self):
        ## Check the time step is existed in title
        if self.all_compute.existed:
            pre_step = EasyStepTag('pre', None, self.init_time)
            for item in self.all_compute.components:
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                if re.search(self.compute_transpose_in.pattern, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.compute_transpose_in.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
                if re.search(self.compute_transpose_out.pattern, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.compute_transpose_out.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
                        self.compute_transpose_out.pre_step = copy.copy(pre_step)
                if item.__contains__('ts') and item.__contains__('dur'):
                    pre_step.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
            
        if self.memcpy.existed:
            for item in self.memcpy.components:
                if not item.__contains__('args') or not item['args'].__contains__('op'):
                    continue
                if re.search(self.memcpyH2D.pattern, item['args']['op'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.memcpyH2D.setup_pid_and_time(self.memcpy.pid, item['ts'], item['dur'])
                if re.search(self.memcpyD2H.pattern, item['args']['op'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.memcpyD2H.setup_pid_and_time(self.memcpy.pid, item['ts'], item['dur'])
        
        if self.replica_cpu.existed:
            for item in self.replica_cpu.components:
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                if re.search(self.retval.pattern, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.retval.setup_pid_and_time(self.replica_cpu.pid, item['ts'], item['dur'])

    def __str__(self):
        tmp_str  = "[Init time] {}\n".format(self.init_time)
        tmp_str += self.replica_cpu.__str__()
        tmp_str += self.all_compute.__str__()
        tmp_str += self.memcpy.__str__()
        tmp_str += self.compute_transpose_in.__str__()
        tmp_str += self.compute_transpose_out.__str__()
        tmp_str += self.memcpyH2D.__str__()
        tmp_str += self.memcpyD2H.__str__()
        tmp_str += self.retval.__str__()
        return tmp_str

class RecordersTrain(object):
    def __init__(self, json_filename = None, json_data = None,
                str_all_compute    = '(GPU:0)*(all Compute)',
                str_memcpy         = '(memcpy)+ (Compute)+',
                str_compute_transpose_in   = 'TransposeNHWCToNCHW',
                str_compute_transpose_out  = 'TransposeNCHWToNHWC',
                str_memcpyH2D      = 'MEMCPYHtoD',
                str_first_calculate = 'sub',
                str_last_calculate  = 'Neg'):
        self.data      = dict()
        self.fail_data = False
        self.eztags = EasyTagsTrain(json_filename, json_data, str_all_compute, str_memcpy,
                    str_compute_transpose_in, str_compute_transpose_out,
                    str_memcpyH2D, str_first_calculate, str_last_calculate)
        self.init_time = self.eztags.init_time

        self.first_gpu_step  = EasyStepTag('first_forward_gpu', None, self.init_time, start_time = sys.maxsize)
        self.last_gpu_step   = EasyStepTag('last_gpu',  None, self.init_time, start_time = 0)
        self.list_before_compute_transpose_out  = list()

        self.transoutInCal   = 0
          
        self.inputs_pattern   = 'inputs' ###TBD for another system 
        if self.eztags.all_compute.existed:
            for item in self.eztags.all_compute.components:
                if item.__contains__('ts') and item.__contains__('dur'):
                    time_now = item['ts'] - self.init_time
                    if time_now < self.first_gpu_step.start_time:
                        if item['args'].__contains__('name') and re.search(self.inputs_pattern, item['args']['name'], re.M | re.I): ### TBD
                            self.eztags.memcpyH2D.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                            continue
                        
                        if len(self.eztags.list_compute_transpose_in) > 0:
                            if not re.search(self.eztags.str_compute_transpose_in, item['args']['name'], re.M|re.I):
                                self.first_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                        else:
                            self.first_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                    
                    if time_now >= self.last_gpu_step.start_time:
                        self.last_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
            self.calculate_result()

    def calculate_result(self):
        self.preprocess_time = self.first_gpu_step.start_time
        self.tranposeout_time = 0
        self.tranposein_time  = 0
        self.calculate_time   = 0
        if self.eztags.first_calculate.existed and self.eztags.last_calculate.existed:
            if self.eztags.all_compute.existed:
                for item in self.eztags.all_compute.components:
                    if item.__contains__('ts') and item.__contains__('dur'):
                        time_now = item['ts'] - self.init_time
                        if time_now <= self.eztags.first_calculate.start_time:
                            continue
                        if time_now >= self.eztags.last_calculate.start_time:
                            continue

                        for i in self.eztags.list_compute_transpose_out:
                            if i.start_time == time_now:
                                self.transoutInCal += self.transoutInCal + i.end_time - i.last_end
                                i.has_calculated = True
            self.calculate_time = self.eztags.last_calculate.end_time - self.eztags.first_calculate.start_time - self.transoutInCal
        else:
            self.fail_data = True
        if len(self.eztags.list_compute_transpose_out) > 0:
            for i in self.eztags.list_compute_transpose_out:
                self.tranposeout_time = self.tranposeout_time + i.end_time - i.last_end     
                if not i.has_calculated:
                    i.has_calculated = True

        if len(self.eztags.list_compute_transpose_in) > 0:
            for i in self.eztags.list_compute_transpose_in:
                if self.first_gpu_step.start_time < i.start_time:
                    self.tranposein_time  = self.tranposein_time  + i.end_time - i.last_end
                    if not i.has_calculated:
                        i.has_calculated = True

        self.execution_time = self.last_gpu_step.end_time - self.first_gpu_step.start_time - self.tranposeout_time - self.tranposein_time - self.calculate_time 
        self.forward_time = self.eztags.first_calculate.start_time - self.first_gpu_step.start_time
        self.backward_time = self.execution_time - self.forward_time
        self.sess_time = self.last_gpu_step.end_time
        self.data = {
            'hashkey':         os.path.splitext(os.path.basename(self.eztags.json_filename))[0],
            'preprocess_time': self.preprocess_time,
            'execution_time':  self.execution_time,
            'forward_time':  self.forward_time,
            'backward_time':  self.backward_time,
            'calculate_time':  self.calculate_time,
            'tranposeout_time': self.tranposeout_time,
            'tranposein_time': self.tranposein_time,
            'sess_time':       self.sess_time
        }

    def __str__(self):
        tmp_str  = self.eztags.__str__()
        tmp_str += self.first_gpu_step.__str__()
        tmp_str += self.last_gpu_step.__str__()
        tmp_str += "*" * 40 + "\n" + "[Result]\n"
        tmp_str += "Fail Data:        {}\n".format(self.fail_data)
        tmp_str += "tranposein_time:  {} ms\n".format(self.tranposein_time/1000)
        tmp_str += "tranposeout_time: {} ms\n".format(self.tranposeout_time/1000)
        tmp_str += "preprocess:       {} ms\n".format(self.preprocess_time/1000)
        tmp_str += "forward_time:     {} ms\n".format(self.forward_time/1000)
        tmp_str += "backward_time:    {} ms\n".format(self.backward_time/1000)
        tmp_str += "execution_time:   {} ms\n".format(self.execution_time/1000)
        tmp_str += "calculate_time:   {} ms\n".format(self.calculate_time/1000)
        tmp_str += "session time      {} ms".format(self.sess_time/1000)
        return tmp_str

class EasyTagsTrain(object):
    """"Tags of all important process name for training"""
    def __init__(self, json_filename = None, json_data = None, 
                    str_all_compute    = '(GPU:0)*(all Compute)',
                    str_memcpy         = '(memcpy)+ (Compute)+',
                    str_compute_transpose_in   = 'TransposeNHWCToNCHW',
                    str_compute_transpose_out  = 'TransposeNCHWToNHWC',
                    str_memcpyH2D      = 'MEMCPYHtoD',
                    str_first_calculate = 'sub',
                    str_last_calculate  = 'Neg'):

        self.init_time  = sys.maxsize
        self.json_filename = json_filename
        self.json_data  = json_data
        self.str_compute_transpose_in   = str_compute_transpose_in
        self.str_compute_transpose_out  = str_compute_transpose_out
        self.str_memcpyH2D = str_memcpyH2D
        self.str_first_calculate = str_first_calculate
        self.str_last_calculate  = str_last_calculate

        self.all_compute   = EasyTagTitle('all_compute', str_all_compute)
        self.memcpy        = EasyTagTitle('memcpy', str_memcpy)
        self.list_compute_transpose_in  = list()
        self.list_compute_transpose_out = list()
        self.memcpyH2D       = None 
        self.first_calculate = None
        self.last_calculate  = None
        ### TBD for Conv2DBackpropInput 
        self.quick_reset()
        self.setup_steps()

    
    def init_json_data(self):
        if not self.json_data:
            if not self.json_filename:
                raise FileNotFoundError("Json Data Not Found!!")
            with open(self.json_filename, 'r') as f:
                self.json_data = json.load(f)
         
    def quick_reset(self):
        if not self.json_data:
            self.init_json_data()
        # Check all title and set init time 
        for item in self.json_data['traceEvents']:
            if item.__contains__('ts') and item['ts'] < self.init_time:
                self.init_time = item['ts'] 
            if item.__contains__('name') and item['name'] == 'process_name':
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                if re.search(self.all_compute.pattern, item['args']['name'], re.M|re.I):
                    self.all_compute.pid = item['pid']
                    self.all_compute.components = self.json_data
                if re.search(self.memcpy.pattern, item['args']['name'], re.M|re.I):
                    self.memcpy.pid = item['pid']
                    self.memcpy.components = self.json_data
                
        self.memcpyH2D       = EasyStepTag('memcpyH2D', self.str_memcpyH2D, self.init_time)
        self.first_calculate = EasyStepTagTrace('First gradients (Sub)', self.str_first_calculate, self.init_time)
        self.last_calculate  = EasyStepTag('Last  gradients (Neg)', self.str_last_calculate, self.init_time)
     
        del self.json_data
    
    def setup_steps(self):
        ## Check the time step is existed in title 
        if self.all_compute.existed:
            pre_step = EasyStepTag('pre', None, self.init_time)
            for item in self.all_compute.components:
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                
                if re.search(self.str_compute_transpose_in, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        tmp = (EasyStepTagTrace('compute_transpose_in_' + str(len(self.list_compute_transpose_in)+1), 
                            self.str_compute_transpose_in, self.init_time))
                        tmp.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
                        tmp.pre_step = copy.copy(pre_step)
                        self.list_compute_transpose_in.append(tmp)

                if re.search(self.first_calculate.pattern, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        if not self.first_calculate.existed and \
                            not re.search(self.str_compute_transpose_in, item['args']['name'], re.M|re.I):
                            self.first_calculate.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
                            self.first_calculate.pre_step = copy.copy(pre_step)
                
                if re.search(self.last_calculate.pattern, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        if not self.last_calculate.existed:
                            self.last_calculate.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])

                if re.search(self.str_compute_transpose_out, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        tmp = (EasyStepTagTrace('compute_transpose_out_' + str(len(self.list_compute_transpose_out)+1), 
                            self.str_compute_transpose_out, self.init_time))
                        tmp.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
                        tmp.pre_step = copy.copy(pre_step)
                        self.list_compute_transpose_out.append(tmp)
                if item.__contains__('ts') and item.__contains__('dur'):
                    pre_step.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
            
        if self.memcpy.existed:
            for item in self.memcpy.components:
                if not item.__contains__('args') or not item['args'].__contains__('op'):
                    continue
                if re.search(self.memcpyH2D.pattern, item['args']['op'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.memcpyH2D.setup_pid_and_time(self.memcpy.pid, item['ts'], item['dur'])
               
    def __str__(self):
        tmp_str  = "[Init time] {}\n".format(self.init_time)
        tmp_str += self.all_compute.__str__()
        tmp_str += self.memcpy.__str__()
        for i in self.list_compute_transpose_in:
            tmp_str += i.__str__()
        tmp_str += self.first_calculate.__str__()
        tmp_str += self.last_calculate.__str__()       
        for i in self.list_compute_transpose_out:
            tmp_str += i.__str__()
        tmp_str += self.memcpyH2D.__str__()
        return tmp_str

class RecordersRT(object):
    """ "Store Data infos """
    def __init__(self, json_filename = None, json_data = None,
                str_replica_cpu = '(replica:0)*(CPU:0)+ (Compute)+', 
                str_all_compute    = '(GPU:0)*(all Compute)',
                str_memcpy         = '(memcpy)+ (Compute)+',
                str_trt_transpose_in   = 'NWith',
                str_memcpyH2D      = 'MEMCPYHtoD',
                str_memcpyD2H      = 'MEMCPYDtoH',
                str_retval         = 'retval'):
        self.data      = dict()
        self.fail_data = False
        self.eztags = EasyTagsRT(json_filename, json_data, str_replica_cpu, str_all_compute, str_memcpy,
                    str_trt_transpose_in,
                    str_memcpyH2D, str_memcpyD2H, str_retval)
        self.init_time = self.eztags.init_time
        ### Search First and Last GPU Step
        self.first_gpu_step = EasyStepTag('first_gpu', None, self.init_time, start_time = sys.maxsize)
        self.last_gpu_step  = EasyStepTag('last_gpu',  None, self.init_time, start_time = 0)

        if not self.eztags.trt_sucess:
            self.fail_data = True
            #return
        if len(self.eztags.list_trt_transpose_in) > 0:
            pre_start = -1
            for i, ele in enumerate(self.eztags.list_trt_transpose_in):
                if i > 1:
                    break
                if pre_start == ele.pre_step.start_time:
                    self.fail_data = True
                    #print("AAAAA", "pre is ", pre_start, "ele_prep is ", ele.pre_step.start_time)
                pre_start = ele.start_time
    
        #pre_step = EasyStepTag('pre_step', None, self.init_time, start_time = sys.maxsize)
        if self.eztags.all_compute.existed:
            for item in self.eztags.all_compute.components:
                if item.__contains__('ts') and item.__contains__('dur'):
                    time_now = item['ts'] - self.init_time              
                    if time_now < self.first_gpu_step.start_time:
                        if len(self.eztags.list_trt_transpose_in) > 0:  # if self.eztags.trt_transpose_in.existed:
                            if not re.search(self.eztags.str_trt_transpose_in, item['args']['name'], re.M|re.I):
                               self.first_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                        else:
                            self.first_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                    
                    if time_now >= self.last_gpu_step.start_time:
                        self.last_gpu_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])

                    #### pre_step setting 
                    #pre_step.setup_pid_and_time(self.eztags.all_compute.pid, item['ts'], item['dur'])
                    #if self.eztags.trt_transpose_in.existed:
                    #    if re.search(self.eztags.str_trt_transpose_in, item['args']['name'], re.M|re.I):
                    #        if pre_step == 'trt_in':
                    #            self.fail_data = True # 2 connecting NWith step means failed data
                    #        pre_step.name = 'trt_in'
                    #    else:
                    #        pre_step = 'pre_step' # reset 
                    #else:
                    #    pre_step = 'pre_step' # reset 

        self.calculate_result()

    def calculate_result(self):
        self.preprocess_time = self.first_gpu_step.start_time

        if len(self.eztags.list_trt_transpose_in) > 1:
            self.execution_time  =  self.eztags.list_trt_transpose_in[1].start_time - self.first_gpu_step.start_time
            #print("QQ1", self.execution_time)
        else:
            self.execution_time  = self.first_gpu_step.wall_time
            #print("QQ2", self.execution_time)


        if self.eztags.memcpyD2H.existed:
            self.memcpyD2H_time  = self.eztags.memcpyD2H.end_time - self.last_gpu_step.end_time
            self.retval_time     = self.eztags.retval.end_time - self.eztags.memcpyD2H.end_time
        else:
            self.memcpyD2H_time = 0
            self.retval_time    = self.eztags.retval.end_time - self.last_gpu_step.end_time
        self.retval_half_time = self.retval_time / 2 

        min_post_start_time = min(self.eztags.retval.start_time, self.eztags.memcpyD2H.start_time)
        if min_post_start_time < self.last_gpu_step.end_time:
            self.fail_data = True
        self.sess_time = max(self.eztags.retval.end_time, self.eztags.memcpyD2H.end_time, self.last_gpu_step.end_time) ### TBD
        self.data = {
            'hashkey':            os.path.splitext(os.path.basename(self.eztags.json_filename))[0],
            'preprocess_time':    self.preprocess_time,
            'execution_time':     self.execution_time,
            'memcpy_time':        self.memcpyD2H_time, 
            'retval_time':        self.retval_time,
            'retval_half_time':   self.retval_half_time,
            'memcpy_retval':      self.memcpyD2H_time + self.retval_time,
            'memcpy_retval_half': self.memcpyD2H_time + self.retval_half_time,
            'sess_time':          self.sess_time
        }
    
    def __str__(self):
        tmp_str  = self.eztags.__str__()
        tmp_str += self.first_gpu_step.__str__()
        tmp_str += self.last_gpu_step.__str__()
        tmp_str += "*" * 40 + "\n" + "[RT-Result]\n"
        tmp_str += "Fail Data:        {}\n".format(self.fail_data)
        tmp_str += "preprocess:       {} ms\n".format(self.preprocess_time/1000)
        tmp_str += "execution_time:   {} ms\n".format(self.execution_time/1000)
        tmp_str += "memcpyD2H_time:   {} ms\n".format(self.memcpyD2H_time/1000)
        tmp_str += "session time      {} ms".format(self.sess_time/1000)
        return tmp_str


class EasyTagsRT(object):
    """"Tags of all important process name"""
    def __init__(self, json_filename = None, json_data = None, 
                    str_replica_cpu = '(replica:0)*(CPU:0)+ (Compute)+', 
                    str_all_compute    = '(GPU:0)*(all Compute)',
                    str_memcpy         = '(memcpy)+ (Compute)+',
                    str_trt_transpose_in   = 'NWith',
                    str_memcpyH2D      = 'MEMCPYHtoD',
                    str_memcpyD2H      = 'MEMCPYDtoH',
                    str_retval         = 'retval'):

        self.init_time  = sys.maxsize
        self.json_filename = json_filename
        self.json_data  = json_data
        self.str_trt_transpose_in   = str_trt_transpose_in
        self.str_memcpyH2D = str_memcpyH2D
        self.str_memcpyD2H = str_memcpyD2H
        self.str_retval    = str_retval
        self.replica_cpu   = EasyTagTitle('replica_cpu', str_replica_cpu)
        self.all_compute   = EasyTagTitle('all_compute', str_all_compute)
        self.memcpy        = EasyTagTitle('memcpy', str_memcpy)
        self.list_trt_transpose_in  = list()
        #self.compute_transpose_in  = None
        #self.compute_transpose_out = None
        self.memcpyH2D     = None
        self.memcpyD2H     = None
        self.retval        = None
        self.trt_sucess    = False
        self.quick_reset()
        self.setup_steps()

    def init_json_data(self):
        if not self.json_data:
            if not self.json_filename:
                raise FileNotFoundError("Json Data Not Found!!")
            with open(self.json_filename, 'r') as f:
                self.json_data = json.load(f)

    def reset_init_time(self):
        for item in self.json_data['traceEvents']:
            # The smallest time as the start time
            if item.__contains__('ts') and item['ts'] < self.init_time:
                self.init_time = item['ts']

    def quick_reset(self):
        if not self.json_data:
            self.init_json_data()
        # Check all title and set init time 
        for item in self.json_data['traceEvents']:
            if item.__contains__('ts') and item['ts'] < self.init_time:
                self.init_time = item['ts'] 
            if item.__contains__('name') and re.search('TRTEngineOp', item['name'], re.M|re.I):
                self.trt_sucess = True
            if item.__contains__('name') and item['name'] == 'process_name':
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                if re.search(self.all_compute.pattern, item['args']['name'], re.M|re.I):
                    self.all_compute.pid = item['pid']
                    self.all_compute.components = self.json_data
                if re.search(self.memcpy.pattern, item['args']['name'], re.M|re.I):
                    self.memcpy.pid = item['pid']
                    self.memcpy.components = self.json_data
                if re.search(self.replica_cpu.pattern, item['args']['name'], re.M|re.I):
                    self.replica_cpu.pid = item['pid']
                    self.replica_cpu.components = self.json_data

        #self.trt_transpose_in  = EasyStepTag('trt_transpose_in',  self.str_trt_transpose_in, self.init_time)
        self.memcpyH2D     = EasyStepTag('memcpyH2D', self.str_memcpyH2D, self.init_time)
        self.memcpyD2H     = EasyStepTag('memcpyD2H', self.str_memcpyD2H, self.init_time)
        self.retval        = EasyStepTag('retval', self.str_retval, self.init_time)
        del self.json_data
    
    def setup_steps(self):
        ## Check the time step is existed in title
        if self.all_compute.existed:
            pre_step = EasyStepTag('pre', None, self.init_time)
            for item in self.all_compute.components:
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                #if re.search(self.trt_transpose_in.pattern, item['args']['name'], re.M|re.I):
                #    if item.__contains__('ts') and item.__contains__('dur'):
                #        self.trt_transpose_in.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
                if re.search(self.str_trt_transpose_in, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        tmp = (EasyStepTagTrace('trt_transpose_in_' + str(len(self.list_trt_transpose_in)+1), 
                            self.str_trt_transpose_in, self.init_time))
                        tmp.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
                        tmp.pre_step = copy.copy(pre_step)
                    
                        self.list_trt_transpose_in.append(tmp)

                
                if item.__contains__('ts') and item.__contains__('dur'):
                    pre_step.setup_pid_and_time(self.all_compute.pid, item['ts'], item['dur'])
            

            
        if self.memcpy.existed:
            for item in self.memcpy.components:
                if not item.__contains__('args') or not item['args'].__contains__('op'):
                    continue
                if re.search(self.memcpyH2D.pattern, item['args']['op'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.memcpyH2D.setup_pid_and_time(self.memcpy.pid, item['ts'], item['dur'])
                if re.search(self.memcpyD2H.pattern, item['args']['op'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.memcpyD2H.setup_pid_and_time(self.memcpy.pid, item['ts'], item['dur'])
        
        if self.replica_cpu.existed:
            for item in self.replica_cpu.components:
                if not item.__contains__('args') or not item['args'].__contains__('name'):
                    continue
                if re.search(self.retval.pattern, item['args']['name'], re.M|re.I):
                    if item.__contains__('ts') and item.__contains__('dur'):
                        self.retval.setup_pid_and_time(self.replica_cpu.pid, item['ts'], item['dur'])
    def __str__(self):
        tmp_str  = "[RT - Init time] {}\n".format(self.init_time)
        tmp_str += self.replica_cpu.__str__()
        tmp_str += self.all_compute.__str__()
        tmp_str += self.memcpy.__str__()
        #tmp_str += self.trt_transpose_in.__str__()
        for i in self.list_trt_transpose_in:
            tmp_str += i.__str__()
        tmp_str += self.memcpyH2D.__str__()
        tmp_str += self.memcpyD2H.__str__()
        tmp_str += self.retval.__str__()
        return tmp_str

class EasyTag(object):
    def __init__(self, name, pattern):
        self._name = name
        self._pattern = pattern
        self._pid     = None
        self._existed = False
    
    @property
    def pid(self):
        return self._pid
    
    @property
    def existed(self):
        return self._existed
    
    @property
    def name(self):
        return self._name
    
    @property
    def pattern(self):
        return self._pattern

    @pid.setter
    def pid(self, pid):
        self._pid = int(pid)
        self._existed = True

    @pattern.setter
    def pattern(self, pattern):
        self._pattern = pattern
    
    @name.setter
    def name(self, name):
        self._name = name

    def __str__(self):
        return "[{}] existed: {}, pid: {}\n".format(self.name, self.existed, self.pid)

class EasyTagTitle(EasyTag):
    def __init__(self, name, pattern):
        self._components = None
        super().__init__(name, pattern)
       
    @property
    def components(self):
        return self._components
    
    @components.setter
    def components(self, json_data):
        self._components = sorted([i for i in json_data['traceEvents'] if i['pid'] == self.pid], 
            key=lambda k: k.get('ts', 0))
            
class EasyStepTag(EasyTag):
    def __init__(self, name, pattern, init_time, start_time = 0, wall_time = 0):
        self._init_time  = int(init_time)
        self._start_time = int(start_time)
        self._wall_time  = int(wall_time)
        super().__init__(name, pattern)
    
    def setup_pid_and_time(self, pid, start_time, wall_time):
        self._start_time = start_time - self.init_time
        self._wall_time  = wall_time
        self.pid = pid
    
    @property
    def init_time(self):
       return self._init_time

    @property
    def start_time(self):
        return self._start_time
    
    @property
    def wall_time(self):
        return self._wall_time
    
    @property
    def end_time(self):
        return self._start_time + self._wall_time
    
    def __str__(self):
        return "[{}] existed: {}, pid: {}, start: {}, wall: {}, end: {}\n".format(self.name, 
            self.existed, self.pid, self.start_time, self.wall_time, self.end_time)

class EasyStepTagTrace(EasyStepTag):
    def __init__(self, name, pattern, init_time, start_time = 0, wall_time = 0, pre_step = None):
        self._last_start = -1
        self._last_wall  = -1
        self._last_end   = -1
        self._pre_step   = pre_step
        self._has_calculated = False
        super().__init__(name, pattern, init_time, start_time, wall_time)
    
    @property
    def pre_step(self):
        return self._pre_step

    @property
    def has_calculated(self):
        return self._has_calculated

    @pre_step.setter
    def pre_step(self, pre_step):
        self._pre_step = pre_step

    @has_calculated.setter
    def has_calculated(self, has_calculated):
        self._has_calculated = has_calculated

    @property
    def last_start(self):
        if self.pre_step and self.pre_step.existed:
            self._last_start = self.pre_step.start_time
            return self._last_start 
    
    @property
    def last_wall(self):
        if self.pre_step and self.pre_step.existed:
            self._last_wall = self.pre_step.wall_time
            return self._last_wall 
    
    @property
    def last_end(self):
        if self.pre_step and self.pre_step.existed:
            return self.last_start + self.last_wall 
    
    def __str__(self):
        return "[{}] existed: {}, pid: {}, start: {}, wall: {}, end: {}  ||  last(s,w): ({},{}), Caled: {}\n".format(self.name, 
            self.existed, self.pid, self.start_time, self.wall_time, self.end_time, self.last_start, self.last_wall, self.has_calculated)

def Recorders(json_filename = None, json_data = None,
                str_replica_cpu = '(replica:0)*(CPU:0)+ (Compute)+', 
                str_all_compute    = '(GPU:0)*(all Compute)',
                str_memcpy         = '(memcpy)+ (Compute)+',
                str_compute_transpose_in   = 'TransposeNHWCToNCHW',
                str_compute_transpose_out  = 'TransposeNCHWToNHWC',
                str_memcpyH2D      = 'MEMCPYHtoD',
                str_memcpyD2H      = 'MEMCPYDtoH',
                str_retval         = 'retval',
                str_first_calculate= 'sub',
                str_last_calculate = 'Neg',
                str_trt_transpose_in = 'NWith',
                is_train = 0,
                is_trt = 0):
    if is_train:
        return  RecordersTrain(json_filename = json_filename, json_data = json_data,
                str_all_compute    = str_all_compute,
                str_memcpy         = str_memcpy,
                str_compute_transpose_in   = str_compute_transpose_in,
                str_compute_transpose_out  = str_compute_transpose_out,
                str_memcpyH2D       = str_memcpyD2H,
                str_first_calculate = str_first_calculate,
                str_last_calculate  = str_last_calculate)
    if is_trt:
        return RecordersRT(json_filename = json_filename, json_data = json_data, 
                str_replica_cpu = str_replica_cpu,
                str_all_compute = str_all_compute, str_memcpy = str_memcpy,
                str_trt_transpose_in  = str_trt_transpose_in, 
                str_memcpyH2D = str_memcpyH2D, str_memcpyD2H = str_memcpyD2H, 
                str_retval = str_retval)

    return RecordersTesting(json_filename = json_filename, json_data = json_data, 
                str_replica_cpu = str_replica_cpu,
                str_all_compute = str_all_compute, str_memcpy = str_memcpy,
                str_compute_transpose_in  = str_compute_transpose_in, 
                str_compute_transpose_out = str_compute_transpose_out,
                str_memcpyH2D = str_memcpyH2D, str_memcpyD2H = str_memcpyD2H, 
                str_retval = str_retval)