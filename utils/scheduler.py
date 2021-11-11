import json
from abc import ABCMeta, abstractmethod
from .utils import _logger

__all__ = ['FixFreqScheduler', 'ListRelativeScheduler', 'get_scheduler']

class Scheduler(metaclass=ABCMeta):
    def __init__(self, lr, json_file):
        self._baselr = lr
        self._last_lr = None
        self._cur_lr  = lr
        self._json    = json_file
        self._scheduler_name = None
        _logger.info("[LRA] Initialize the Learning rate to {}".format(self._cur_lr))
    
    @abstractmethod
    def adjust_lr(self, epoch):
        pass

    @abstractmethod
    def need_update(self, epoch):
        pass

    @property
    def baselr(self):
        return self._baselr
    
    @property
    def cur_lr(self):
        return self._cur_lr
    
    @property
    def scheduler_name(self):
        return self._scheduler_name

class FixFreqScheduler(Scheduler):
    def __init__(self, lr, json_file):
        super(FixFreqScheduler, self).__init__(lr, json_file)
        self._scheduler_gamma = None
        self._scheduler_step  = None
        if self._json:
            self._getfromjson()
    
    def force_set_scheduler(self, step, gamma):
        self._scheduler_step = step
        self._scheduler_gamma = gamma
        
    def _getfromjson(self):
        with open(self._json) as f:
            data = json.load(f)
        self._scheduler_name  = data['scheduler_name']
        self._scheduler_step  = data['scheduler_step']
        self._scheduler_gamma = data['scheduler_gamma']

    def adjust_lr(self, epoch):
        if not self.need_update(epoch):
            return 
        self._last_lr = self._cur_lr
        self._cur_lr  = self._baselr * (self._scheduler_gamma ** (epoch // self._scheduler_step))
        _logger.info("[LRA] Learning rate is updated form {} to {}".format(self._last_lr, self._cur_lr))

    def need_update(self, epoch):
        if epoch == 0 or (epoch % self._scheduler_step) != 0:
            return 0
        return 1

    @property
    def scheduler_gamma(self):
        return self._scheduler_gamma
    
    @property
    def scheduler_step(self):
        return self._scheduler_step

class ListRelativeScheduler(Scheduler):
    def __init__(self, lr, json_file):
        super(ListRelativeScheduler, self).__init__(lr, json_file)
        self._scheduler_list = None
        if self._json:
            self._getfromjson()

    def _getfromjson(self):
        with open(self._json) as f:
            data = json.load(f)
        self._scheduler_name  = data['scheduler_name']
        self._scheduler_step  = data['scheduler_step']
        self._scheduler_gamma = data['scheduler_gamma']
        assert len(self._scheduler_step) == len(self._scheduler_gamma), "len of step should equal to len of gamma"
        self._auto_sort()
        self._sched_index = 0

    def _auto_sort(self):
        ## Step: ascending, Gamma: descending
        self._scheduler_step.sort() 
        self._scheduler_gamma.sort(reverse = True) 

    def adjust_lr(self, epoch):
        if not self.need_update(epoch):
            return
        self._last_lr = self._cur_lr
        self._cur_lr  = self._baselr * self._scheduler_gamma[self._sched_index]
        if self._sched_index < (len(self._scheduler_step) - 1): # Notice: Index isn't over than len of list
            self._sched_index += 1 
        _logger.info("[LRA] Learning rate is updated form {} to {}".format(self._last_lr, self._cur_lr))
    
    def need_update(self, epoch):
        if epoch != self._scheduler_step[self._sched_index]:
            return 0
        return 1
    
    @property
    def scheduler_gamma(self):
        return self._scheduler_gamma[self._sched_index]
    
    @property
    def scheduler_step(self):
        return self._scheduler_step[self._sched_index]

    @property
    def sched_index(self):
        return self._sched_index

def get_scheduler(mode, learning_rate, filename = None, step = None, gamma = None):
    if mode == "":
        sched = FixFreqScheduler(learning_rate, None)
        sched.force_set_scheduler(step, gamma)
    elif mode == "fixfreq":
        sched = FixFreqScheduler(learning_rate, filename)
    elif mode == "listrelative":
        sched = ListRelativeScheduler(learning_rate, filename)
    return sched

