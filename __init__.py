from . import base2 as b
from . import api2 as api
from .api2 import Registery
from .base2 import Rglobal, QTglobal
from .data_parser2 import  parse_metrix_table, parse_convertors_table, parse_kinds_table, parse_units_table
from .data_quantity import (unit_txt_tables , kind_txt_tables, metrix_txt_tables,  convertor_txt_tables)
from .registery import QuantityTypes

for tbl in metrix_txt_tables:
    parse_metrix_table(Rglobal, tbl)    

for tbl in convertor_txt_tables:
    parse_convertors_table(Rglobal, tbl)

for tbl in kind_txt_tables:
    parse_kinds_table(Rglobal, tbl)

for tbl in unit_txt_tables:
    parse_units_table(Rglobal, tbl)

q = QuantityTypes(Registery())

for tbl in metrix_txt_tables:
    parse_metrix_table(q.R, tbl)

# q.make_kind('length', baseunit='m')
# q.make_unit('length', "m", 1.0, metrix=True)
# q.make_kind('mass', baseunit='m')
# q.make_unit('mass', "g", 1.0, metrix=True)

# q.make_kind('time', baseunit="s")
# q.make_unit('time', 's', 1.0)

#for tbl in metrix_txt_tables:
#    parse_metrix_table(q.R, tbl)    
#
#for tbl in convertor_txt_tables:
#    parse_convertors_table(q.R, tbl)
#
#for tbl in kind_txt_tables:
#    parse_kinds_table(q.R, tbl)
#
#for tbl in unit_txt_tables:
#    parse_units_table(q.R, tbl)

try:
    import numpy
except:
    pass
else:   
    from .numpyquantity import *

QTglobal.__update__()

_G_= globals()

for sub in QuantityTypes.__mro__:
    for attr, _ in sub.__dict__.iteritems():
        if attr[0]!="_":
            _G_.setdefault(attr, getattr(QTglobal, attr))

for attr, method in QTglobal.__dict__.iteritems():
    if attr[0]!="_":
        _G_.setdefault(attr, method)

