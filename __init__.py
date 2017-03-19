from . import base as b
from . import api
from .api import Registery
from .base import QTglobal
from .registery import QuantityTypes

try:
    pass
    #import numpy
except:
    pass
else:   
    from .numpyquantity import *
    
QTglobal.register_system_units()

_G_= globals()
for sub in QuantityTypes.__mro__:
    for attr, _ in sub.__dict__.items():
        if attr[0]!="_":
            _G_.setdefault(attr, getattr(QTglobal, attr))
            
for attr, method in QTglobal.__dict__.items():
    if attr[0]!="_":
        _G_.setdefault(attr, method)

del _G_
del sub
del attr
del method

def new_register(empty=False):
    qt = QuantityTypes(Registery())
    if empty:
        qt.register_system_units("metrix")
    else:
        qt.register_system_units()
    return qt
                    
###########################################
## test area 
###########################################

