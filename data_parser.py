import math
from .data_quantity import (K_NAME, K_PYTHON, K_DEFINITION, K_BASE, K_N, K_ULESS,
                            U_NAME,  U_PRINT, U_PYTHON, U_LATEX, U_CI, U_KIND, U_M, U_DIM, U_DEFINITION, U_N,
                            M_NAME, M_PRINT, M_PYTHON, M_LATEX, M_UC, M_SCALE, M_N,
                            C_KINDS, C_TARGETS, C_PYTHON, C_N)


from .api2 import make_unit, make_kind, make_convertor, make_metrix


units_table = []
kinds_table = []
metrix_table = []
convertors_table = []

def parse_table2list(txt, L=None, N=2):
    """ parse a string definition table to a list of columns 

    columns are separated by '|', the dta start after the line '|--', 
    everything before is ignored (header)

    Parameters
    ----------
    txt : string
        The table
    L : list
        list from wich the tble will be happend (modified)
    N : int
        number of columns        
    """
    L = [] if L is None else L
    in_data = False
    for i, line in enumerate(txt.split("\n"), start=1):
        line = line.strip()

        if in_data:
            if not line: continue
            tmp = [v.strip() for v in line.split("|") ][1:-1]


            if len(tmp)!=N:
                print ("WARNING problem reading table at line %d"%i)
            else:    
                L.append(tmp)
        else:
            if line[0:3]=="|--":
                in_data = True


######################################################################################
#
#  Define the Metrix Parsing function.
#
######################################################################################



def parse_metrix_table(R, txt):
    global metrix_table
    tbl = []
    parse_table2list(txt, tbl, N=M_N)
    process_metrix_table(R, tbl)
    metrix_table.extend(tbl)

def process_metrix_table(R, tbl):
    for info in tbl:
        process_metrix(R, info)

def process_metrix(R, info):
    return make_metrix(R, info[M_PYTHON], eval(info[M_SCALE]), name=info[M_NAME], prt=info[M_PRINT], latex=info[M_LATEX])    


######################################################################################
#
#  Define the Kind Parsing function.
#
######################################################################################


def parse_kinds_table(R, txt):
    global kinds_table
    tbl = []
    parse_table2list(txt, tbl, N=K_N)
    process_kinds_table(R, tbl)
    kinds_table.extend(tbl)




def process_kinds_table(R, tbl):
    for info in tbl:
        process_kind(R, info)


def process_kind(R, info):    
    make_kind(R, info[K_PYTHON], info[K_DEFINITION], info[K_BASE], name=info[K_NAME], unitless=info[K_ULESS] == "yes")
   

######################################################################################
#
#  Define the Unit Parsing function.
#
######################################################################################


def parse_units_table(R, txt):
    global units_table
    tbl = []
    parse_table2list(txt, tbl, N=U_N)
    process_units_table(R, tbl)
    units_table.extend(tbl)


def process_units_table(R, tbl):
    for info in tbl:
        process_unit(R, info)



def process_unit(R, info):
    make_unit(R, info[U_PYTHON], info[U_DEFINITION], info[U_KIND], 
                    dimension=int(info[U_DIM] or 1) ,
                    metrix=info[U_M].lower()=="yes", 
                    name=info[U_NAME], 
                    prt=info[U_PRINT]
            )


######################################################################################
#
#  Define the Convertor Parsing function.
#
######################################################################################





def parse_convertors_table(R, txt):
    global convertors_table
    tbl = []
    parse_table2list(txt, tbl, N=C_N)
    process_convertors_table(R, tbl)
    convertors_table.extend(tbl)


def process_convertors_table(R, tbl):
    for info in tbl:
        process_convertor(R, info)

def process_convertor(R, info):    
    make_convertor(R, info[C_KINDS], info[C_TARGETS], info[C_PYTHON])    
             
            

def test():
    txt = """
|                 name                 | print | python | latex |  c/i  |        kind       |  M  | pow |     definition    |
|--------------------------------------|-------|--------|-------|-------|-------------------|-----|-----|-------------------|
| TEST                                 | K     | test   | Ky    | KY    | length     | yes |   1 | m           |
"""
    parse_units_table(txt)
