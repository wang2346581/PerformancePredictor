
import math
import numpy as np

__all__ = ['DExpRegErrorMeter', 'RegErrorMeter', 'AverageMeter', 
    'ErrorMeter', 'ExpErrorMeter', 'ExpRegErrorMeter', 'DExpErrorMeter']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0
        self._correct = 0

    def update(self, val, n=1):
        self._val = val
        self._sum += val * n
        self._count += n
        self._avg = self._sum / self._count

    @property
    def count(self):
        return int(self._count)

    @property
    def avg(self):
        return float(self._avg)
    
    @property
    def correct(self):
        return int(self._sum) // 100


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ErrorMeter(object):
    """Computes and stores the average error and current value"""
    def __init__(self, name, fmt=':f', magic_scaler = 1):
        self.name = name
        self.fmt = fmt
        self.magic_scaler = magic_scaler
        self.reset()

    def reset(self):
        self._calculated   = False
        self._mean_of_pred = 0
        self._sse_sum      = 0
        self._sst_sum      = 0
        self._ans_Sub_pred = None
        self._meanans_Sub_ans = None
        self._ans  = np.array([])
        self._pred = np.array([])
    
    def update(self, ans, pred):
        if not isinstance(ans, np.ndarray):
            ans =  np.array(ans)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)
        assert self._ans.shape == self._pred.shape, 'len of pred should be same as len of ans'
        
        if self.magic_scaler != 1 and self.magic_scaler != 0:
            ans  = ans  / self.magic_scaler
            pred = pred / self.magic_scaler

        self._ans  = np.append(self._ans, ans)
        self._pred = np.append(self._pred, pred)
        
    
    def summary(self):
        assert not self._calculated, 'Do not calculate 2 twice, only do this function at end of epochs'
        self._ans_Sub_pred = np.subtract(self._ans, self._pred)
        self._mean_of_ans = np.mean(self._ans)
        # self._meanpred_Sub_pred = np.square(self._mean_of_pred - self._ans)
        self._meanans_Sub_ans = np.square(self._mean_of_ans - self._ans)
        self._sse_sum = np.sum(np.square(self._ans_Sub_pred))
        self._sst_sum = np.sum(self._meanans_Sub_ans)
        self._calculated = True

    @property
    def prediction(self):
        return self._pred

    @property
    def answer(self):
        return self._ans

    @property
    def count(self):
        return len(self._ans)

    @property
    def mae(self):
        return np.mean(np.absolute(self._ans_Sub_pred))

    @property
    def mape(self):
        return np.mean(np.divide(np.absolute(self._ans_Sub_pred), np.array(self._ans))) * 100

    @property
    def rmse(self):
        return np.sqrt(self._sse_sum / self.count)

    @property
    def r2(self):
        return float(1 - (self._sse_sum / self._sst_sum))
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ExpErrorMeter(ErrorMeter):
    """Computes and stores the average error and current value"""
    def __init__(self, name, fmt=':f', magic_scaler = 1):
        super(ExpErrorMeter, self).__init__(name, fmt=':f', magic_scaler = magic_scaler)
    
    def update(self, ans, pred):
        if not isinstance(ans, np.ndarray):
            ans =  np.array(ans)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)
        assert self._ans.shape == self._pred.shape, 'len of pred should be same as len of ans'

        if self.magic_scaler != 1 and self.magic_scaler != 0:
            pred = np.exp(pred) / self.magic_scaler 
        
        self._ans  = np.append(self._ans, ans)
        self._pred = np.append(self._pred, pred)


class DExpErrorMeter(ErrorMeter):
    """Computes and stores the average error and current value"""
    def __init__(self, name, fmt=':f', magic_scaler = 1):
        super(DExpErrorMeter, self).__init__(name, fmt=':f', magic_scaler = magic_scaler)
    
    def update(self, ans, pred):
        if not isinstance(ans, np.ndarray):
            ans =  np.array(ans)
        if not isinstance(pred, np.ndarray):
            pred = np.array(pred)
        assert self._ans.shape == self._pred.shape, 'len of pred should be same as len of ans'
        
        if self.magic_scaler != 1 and self.magic_scaler != 0:
            pred = np.exp(pred) / self.magic_scaler
            ans  = np.exp(ans)  / self.magic_scaler 

        self._ans  = np.append(self._ans, ans)
        self._pred = np.append(self._pred, pred)

class RegErrorMeter(object):
    def __init__(self, magic_scaler = 1):
        self.loss  = AverageMeter('loss', ':.4f')
        self.error = ErrorMeter('error', ':6.2f', magic_scaler = magic_scaler)
    
class ExpRegErrorMeter(object):
    def __init__(self, magic_scaler = 1):
        self.loss  = AverageMeter('loss', ':.4f')
        self.error = ExpErrorMeter('experror', ':6.2f', magic_scaler = magic_scaler)

class DExpRegErrorMeter(object):
    def __init__(self, magic_scaler = 1):
        self.loss  = AverageMeter('loss', ':.4f')
        self.error = DExpErrorMeter('Dexperror', ':6.2f', magic_scaler = magic_scaler)