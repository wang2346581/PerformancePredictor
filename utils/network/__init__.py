from .perfnetA import *
from .perfnetB import *
from .perfnetAD import *
from .perfnetAS import *
from .perfnetA1 import *
from .perfnetA2 import *
from .perfnetA3 import *
from .perfnetA5 import *
from .perfnetADan import *

import os
from pkgutil import iter_modules

__all__ = []

def _global_import(name, all_list):
    p = __import__(name, globals(), locals(), level=1)
    lst = p.__all__ if '__all__' in dir(p) else []
    if lst:
        del globals()[name]
        for k in lst:
            globals()[k] = p.__dict__[k]
            all_list.append(k)

_CURR_DIR = os.path.dirname(__file__)
for _, module_name, _ in iter_modules([_CURR_DIR]):
    srcpath = os.path.join(_CURR_DIR, module_name + '.py')
    if not os.path.isfile(srcpath):
        continue
    if module_name.startswith('_'):
        continue
    _global_import(module_name, __all__)

def get_local_nn(inputs, training, network_name="perfnetA", pred = None):
    nn_map = {
        'perfnetA': perfnetA(inputs, training),
        'perfnetAr45d35': perfnetAr45d35(inputs, training),
        'perfnetB': perfnetB(inputs, training),
        'perfnetAD': perfnetAD(inputs, training),
        'perfnetAS': perfnetAS(inputs, training),
        'perfnetA1': perfnetA1(inputs, training),
        'perfnetA2': perfnetA2(inputs, training),
        'perfnetA3': perfnetA3(inputs, training),
        'perfnetA5': perfnetA5(inputs, training),
        'perfnetADan': perfnetADan(inputs, training),
    }
    return nn_map[network_name]

def get_nn_list():
    return _global_import(module_name, [])

__all__.append('get_local_nn')
