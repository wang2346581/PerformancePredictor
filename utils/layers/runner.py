import os
import time
import json
import psutil
import signal
import numpy as np
import pandas as pd
from scipy import stats
from termcolor import colored
from multiprocessing import Process
import humanfriendly
class RunnerBase():
    def __init__(self, memcopyin, output_path, iter_warmup, iter_benchmark, hashkey, **kwargs):
        self._memcopyin      = memcopyin
        self._iter_warmup    = iter_warmup
        self._iter_benchmark = iter_benchmark
        self._hashkey        = hashkey
        self._output_path    = output_path
        self._input          = None
        self._np_input       = None
        self._op             = None
        self._tensor_input_name   = "X" 
        self._tensor_output_name  = None
        self._tensor_output_names = None
        self._slice_second = 0.2
        self._list_time = []
        self._process   = None
        super(RunnerBase, self).__init__(**kwargs)
    
    def gen_input(self):
        raise NotImplementedError("Please Implement this method")
    
    def gen_np_input(self):
        raise NotImplementedError("Please Implement this method")

    def gen_op(self):
        raise NotImplementedError("Please Implement this method")

    def gen_process(self, freeze_mode = 0, exe_mode = 1, profile_mode = 0):
        raise NotImplementedError("Please Implement this method")

    def monitor_process(self):
        raise NotImplementedError("Please Implement this method")
    
    @property
    def memcopyin(self):
        return self._memcopyin
    
    @property
    def output_path(self):
        return self._output_path
    
    @property
    def first_run(self):
        return not os.path.isfile(self.output_path)

    @property
    def iter_warmup(self):
        return self._iter_warmup
    
    @property
    def iter_benchmark(self):
        return self._iter_benchmark
    
    @property
    def hashkey(self):
        return self._hashkey

    @property
    def op(self):
        return self._op
    
    @property
    def tensor_input_name(self):
        return self._tensor_input_name
    
    @property
    def tensor_output_name(self):
        return self._tensor_output_name

    @property
    def tensor_output_names(self):
        if not self._tensor_output_names:
            return [self._tensor_output_name]
        else:
            return self._tensor_output_names

    @property
    def input(self):
        return self._input
    
    @property
    def np_input(self):
        return self._np_input

    @property
    def list_time(self):
        return self._list_time
    
    @property
    def slice_second(self):
        return self._slice_second
    
    @property
    def process(self):
        return self._process

    @op.setter
    def op(self, op):
        self._op = op
    
    @input.setter
    def input(self, inputs):
        self._input = inputs
    
    @np_input.setter
    def np_input(self, np_input):
        self._np_input = np_input
    
    @list_time.setter
    def list_time(self, list_time):
        self._list_time = list_time
    
    @tensor_input_name.setter
    def tensor_input_name(self, tensor_input_name):
        self._tensor_input_name = tensor_input_name

    @tensor_output_name.setter
    def tensor_output_name(self, tensor_output_name):
        self._tensor_output_name = tensor_output_name

    @process.setter
    def process(self, process):
        self._process = process

class Runner(RunnerBase):
    def __init__(self, memcopyin, output_path, iter_warmup, iter_benchmark, hashkey, **kwargs):
        super(Runner, self).__init__(memcopyin=memcopyin, output_path=output_path, 
            iter_warmup=iter_warmup, iter_benchmark=iter_benchmark, 
            hashkey=hashkey, **kwargs)

    def run_without_store(self):
        import tensorflow as tf

        if self.first_run:
            warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
            success_tag = colored('[Success] ', 'green')
            print(success_tag + 'GPU is found') if tf.test.gpu_device_name() else print(warn_tag + 'GPU is Not found')

        tf.reset_default_graph()
        sess = tf.Session()
        self.gen_op()
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        if self.memcopyin:
            self.gen_np_input()
            # Do WarmUp               
            for _ in range(self.iter_warmup):
                sess.run(self.op, feed_dict = {self.input: self.np_input})
            # Do Benchmark
            for _ in range(self.iter_benchmark):
                start_time = time.time()
                sess.run(self.op, feed_dict = {self.input: self.np_input})
                self.list_time.append(((time.time()-start_time) * 1000)) # The unit of measurement is 'Millisecond'
        else:
            for _ in range(self.iter_warmup):
                sess.run(self.op)
            # Do Benchmark
            for _ in range(self.iter_benchmark): 
                start_time = time.time()
                sess.run(self.op)
                self.list_time.append(((time.time()-start_time) * 1000)) # The unit of measurement is 'Millisecond'
        print(self.list_time)
    
    def run_profile(self):
        import tensorflow as tf
        from tensorflow.python.client import timeline

        if self.first_run:
            warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
            success_tag = colored('[Success] ', 'green')
            print(success_tag + 'GPU is found') if tf.test.gpu_device_name() else print(warn_tag + 'GPU is Not found')

        filename = os.path.join(self.output_path, self.hashkey + '.json')
        tf.reset_default_graph()
        run_metadata = tf.RunMetadata()

        self.gen_op()
        sess = tf.Session()
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        if self.memcopyin:
            self.gen_np_input()
            # Do WarmUp
            for _ in range(self.iter_warmup):
                sess.run(self.op, feed_dict = {self.input: self.np_input})
            # Do Benchmark            
            sess.run(self.op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata, feed_dict = {self.input: self.np_input})
        else:
            # Do WarmUp
            for _ in range(self.iter_warmup):
                sess.run(self.op)
            # Do Benchmark            
            sess.run(self.op, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata)
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(filename, 'w') as f:
            f.write(ctf) 
        time.sleep(self.slice_second)
    
    def store(self):
        if not self.list_time:
            print("Pleas run the tensor before storing to csv file")
            return
        time_data_ele = {
            'hashkey':        self.hashkey,
            'time_max':       np.amax(self.list_time),
            'time_min':       np.amin(self.list_time),
            'time_median':    np.median(self.list_time),
            'time_mean':      np.mean(self.list_time), 
            'time_trim_mean': stats.trim_mean(self.list_time, 0.1),
        }
        df_ele = pd.DataFrame(data = time_data_ele, index=[0])
        print("time_mean: {} ms".format(time_data_ele['time_mean']))
        if self.first_run:
            df_ele.to_csv(self.output_path, index=None)
        else:
            df_ele.to_csv(self.output_path, index=False, mode='a', header=False)
        return 
    
    def run_with_store(self):
        self.run_without_store()
        self.store()
        return 
    
    def gen_process(self, freeze_mode = 0, exe_mode = 1, profile_mode = 0):
        if exe_mode:
            self.process = Process(target=self.run_with_store)
        elif profile_mode:
            self.process = Process(target=self.run_profile)
            
    def run_store_on_childprocess(self):
        p = Process(target=self.run_with_store)       
        p.start()
        p.join()
        return 
    
    def run_profile_on_childprocess(self):
        p = Process(target=self.run_profile)       
        p.start()
        p.join()
        return 

    def run_store_with_monitor_childprocess(self):
        p = Process(target=self.run_with_store) 
        p.start()
        self.monitor_process()
        p.join()
        return
    
    def run(self, exe_mode = 1, profile_mode = 0):
        if exe_mode:
            self.run_with_store()
        elif profile_mode:
            self.run_profile()

    def run_on_child(self, freeze_mode = 0, exe_mode = 1, profile_mode = 0, monitor=0):
        self.gen_process(freeze_mode, exe_mode, profile_mode)
        if not self.process:
            print("Not support this function")
            return
        self.process.start()
        if monitor:
           self.monitor_process()
        self.process.join()
        return 
    
    def monitor_process(self):
        #mem_total = psutil.virtual_memory().total
        #mem_per5  = mem_total * 0.05
        pid = self.process.pid
        while self.process.is_alive():
            #mem_free  = psutil.virtual_memory().free
            per_ = psutil.virtual_memory().percent
            #print("====> child pid is {}, mem_per5 is {}, free is {}".format(pid, 
            #    humanfriendly.format_size(mem_per5), 
            #    humanfriendly.format_size(mem_free)))
            #if(mem_free < mem_per5):
            if(psutil.virtual_memory().percent > 95):
                print(humanfriendly.format_size(per_))
                print("Already to OOM, Kill the process!!")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1 * 5)
            time.sleep(5)

class RunnerRT(Runner):
    def __init__(self, memcopyin, output_path, 
        output_pb_path, output_json_path,
        iter_warmup, iter_benchmark, hashkey, **kwargs):
        self._output_pb_path   = output_pb_path
        self._output_json_path = output_json_path
        self._input_str  = 'input'
        self._output_str = 'output'
        self._trtpb_def  = None
        super(RunnerRT, self).__init__(memcopyin=memcopyin, output_path=output_path, 
            iter_warmup=iter_warmup, iter_benchmark=iter_benchmark, 
            hashkey=hashkey, **kwargs)
    
    def get_batchsize(self):
        raise NotImplementedError("Please Implement this method")

    def read_inputs_from_json(self):
        json_filename = os.path.join(self.output_json_path, self.hashkey + '.json')
        with open(json_filename, 'r') as f:
            data = json.load(f)
        self.tensor_input_name  = data[self.input_str]
        self.tensor_output_name = data[self.output_str][0]
    
    def read_pb(self):
        import tensorflow as tf
        pb_filename = os.path.join(self.output_pb_path, self.hashkey + '.pb')
        with tf.gfile.FastGFile(pb_filename, 'rb') as f:
            self.trtpb_def = tf.get_default_graph().as_graph_def()
            self.trtpb_def.ParseFromString(f.read())
    
    def run_without_store(self): ## Unnecessary #TBD
        import tensorflow as tf
        import tensorflow.contrib.tensorrt as trt
        if self.first_run:
            warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
            success_tag = colored('[Success] ', 'green')
            print(success_tag + 'GPU is found') if tf.test.gpu_device_name() else print(warn_tag + 'GPU is Not found')
        
        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                #self.gen_input()
                self.read_inputs_from_json()
                self.read_pb()
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                tf.import_graph_def(self.trtpb_def, name='')
                
                #for op in graph.get_operations():
                #    print(op.name)
                #print(self.tensor_input_name, self.tensor_output_name)
                   
                tf_output = sess.graph.get_tensor_by_name(self.tensor_output_name + ':0')
                if self.memcopyin:
                    tf_input  = sess.graph.get_tensor_by_name(self.tensor_input_name + ":0")
                    self.gen_np_input()
                    #print("---------------------->", self.np_input.shape)
                    # Do WarmUp               
                    for _ in range(self.iter_warmup):
                        sess.run(tf_output, feed_dict = {tf_input: self.np_input})
                    # Do Benchmark
                    for _ in range(self.iter_benchmark):
                        start_time = time.time()
                        sess.run(tf_output, feed_dict = {tf_input: self.np_input})
                        self.list_time.append(((time.time()-start_time) * 1000)) # The unit of measurement is 'Millisecond'
                else:
                    for _ in range(self.iter_warmup):
                        sess.run(tf_output)
                    # Do Benchmark
                    for _ in range(self.iter_benchmark): 
                        start_time = time.time()
                        sess.run(tf_output)
                        self.list_time.append(((time.time()-start_time) * 1000)) # The unit of measurement is 'Millisecond'
                print(self.list_time)

    def run_profile(self):
        import tensorflow as tf
        from tensorflow.python.client import timeline
        import tensorflow.contrib.tensorrt as trt

        if self.first_run:
            warn_tag = colored('[Warn] ', 'red', attrs=['blink']) 
            success_tag = colored('[Success] ', 'green')
            print(success_tag + 'GPU is found') if tf.test.gpu_device_name() else print(warn_tag + 'GPU is Not found')
        
        filename = os.path.join(self.output_path, self.hashkey + '.json')

        with tf.Graph().as_default() as graph:
            with tf.Session(graph=graph) as sess:
                run_metadata = tf.RunMetadata()
                self.read_inputs_from_json()
                self.read_pb()
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                tf.import_graph_def(self.trtpb_def, name='')
                tf_output = sess.graph.get_tensor_by_name(self.tensor_output_name + ':0')
                if self.memcopyin:
                    tf_input  = sess.graph.get_tensor_by_name(self.tensor_input_name + ":0")
                    self.gen_np_input()
                    # Do WarmUp
                    for _ in range(self.iter_warmup):
                        sess.run(tf_output, feed_dict = {tf_input: self.np_input})
                    # Do Benchmark            
                    sess.run(tf_output, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        run_metadata=run_metadata, feed_dict = {tf_input: self.np_input})
                else:
                    # Do WarmUp
                    for _ in range(self.iter_warmup):
                        sess.run(tf_output)
                    # Do Benchmark            
                    sess.run(tf_output, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                        run_metadata=run_metadata)
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                with open(filename, 'w') as f:
                    f.write(ctf) 
                time.sleep(self.slice_second)                
                        
    def freeze_trt_graph(self):
        import tensorflow as tf
        import tensorflow.contrib.tensorrt as trt
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        config.gpu_options.allow_growth = True 
        tf.reset_default_graph() #reset graph 
        
        with tf.Graph().as_default() as graph:
            with tf.Session(config=config, graph=graph) as sess:
                #self.gen_freeze_op()
                self.gen_op()
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                self.tensor_output_name = self.op.name.split(":")[0]
                frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                sess.graph_def, output_node_names = self.tensor_output_names)
                #print("----------------------------->", self.tensor_output_name)
                self.trtpb_def = trt.create_inference_graph(
                    input_graph_def=frozen_graph,
                    max_batch_size=self.get_batchsize(), ### TBD
                    outputs=self.tensor_output_names,
                    max_workspace_size_bytes=(1 << 32),
                    precision_mode='FP32'.upper(),
                    minimum_segment_size= 1,
                )
                #for op in graph.get_operations():
                #    print(op.name)
                #print(self.tensor_output_names, self.tensor_input_name)
                pb_filename = os.path.join(self.output_pb_path, self.hashkey + '.pb')
                with tf.gfile.GFile(pb_filename, "wb") as f:
                    f.write(self.trtpb_def.SerializeToString())
                json_filename = os.path.join(self.output_json_path, self.hashkey + '.json')
                with open(json_filename, 'w') as f:
                    json.dump({
                        self.input_str:  self.tensor_input_name,
                        self.output_str: self.tensor_output_names,
                    }, f)
    
    def gen_process(self, freeze_mode = 0, exe_mode = 1, profile_mode = 0):
        if freeze_mode:
            self.process = Process(target=self.freeze_trt_graph)       
        elif exe_mode:
            self.process = Process(target=self.run_with_store)
        elif profile_mode:
            self.process = Process(target=self.run_profile)

    @property
    def output_pb_path(self):
        return self._output_pb_path

    @property
    def output_json_path(self):
        return self._output_json_path

    @property
    def input_str(self):
        return self._input_str
    
    @property
    def output_str(self):
        return self._output_str
    
    @property
    def trtpb_def(self):
        return self._trtpb_def

    @trtpb_def.setter
    def trtpb_def(self, trtpb_def):
        self._trtpb_def = trtpb_def



