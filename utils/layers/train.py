from utils.utils import get_support_optimizer, get_support_optimizer_function

class TrainCore():
    def __init__(self, sgd, rmsprop, adagrad, adam, **kwargs):
        self._target      = None
        self._opt         = None
        self._opt_sgd     = sgd
        self._opt_adagrad = adagrad
        self._opt_rmsprop = rmsprop
        self._opt_adam    = adam
        super(TrainCore, self).__init__(**kwargs)
    
    def gen_output(self):
        raise NotImplementedError("Please Implement this method")
        
    def gen_op(self):
        raise NotImplementedError("Please Implement this method")

    def gen_opt(self):
        import tensorflow as tf
        opt_function_name = None
        for opt in get_support_optimizer():
            if eval('self.' + opt):
                opt_function_name = get_support_optimizer_function(opt)
                break
        self.opt = eval('tf.train.{}'.format(opt_function_name))
    
    @property
    def target(self):
        return self._target
    
    @property
    def opt(self):
        return self._opt

    @property
    def sgd(self):
        return self._opt_sgd
    
    @property
    def adagrad(self):
        return self._opt_adagrad

    @property
    def rmsprop(self):
        return self._opt_rmsprop
    
    @property
    def adam(self):
        return self._opt_adam
    
    @target.setter
    def target(self, target):
        self._target = target
    
    @opt.setter
    def opt(self, opt):
        self._opt = opt