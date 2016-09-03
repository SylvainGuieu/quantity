import math
from .unitkind import _KindHash_, _KindName_

__all__ = ["kindofunit", "scaleofunit", "unitofunit", "unitofscale", "hashofkind",
            "hashofunit", 
            "unitof", "kindof", "valueof", "scaleof", "isunitless", "unitsofkind",
            "printofunit", "definitionofunit", "nameofunit", "metrixofunit", 
            "baseofunit", "basescaleofunit", "baseofkind", "isunitof", "getkinds",
            "getunits", "arekindconvertible", "areunitconvertible", "linksofkind"
        ]


##
# This is the the global list available when unit or kind defintion 
# are entered as a string 

eval_globals = {"math": math}

class Registery(object):
    __isglobal__ = False
    def __init__(self):
        # Metrix
        self.metrix_lookup = {}
        # Kinds 

        self.kind_lookup = {}

        ## need a fast access from H to kind 
        self.hash2kind_lookup = {}    
        # we need a fast access to wish kind,scale pairs give a unit
        self.scale2unit_lookup = {} 

        # units
        self.unit_lookup = {}    
                
        # convertor
        self.convertor_lookup = {}

        ## save some time to the interpretor build them 
        ## before
        self.interp_kindhash = Interpretor(self,"kindhash").eval
        self.interp_kindhash = Interpretor(self, "kindhash").eval
        self.interp_unithash = Interpretor(self, "unithash").eval
        self.interp_unitlesshash = Interpretor(self, "unitlesshash").eval
        self.interp_unitkindname = Interpretor(self, "unitkindname").eval
        self.interp_kindname = Interpretor(self, "kindname").eval
        self.interp_kindbase = Interpretor(self, "kindbase").eval
        self.interp_scale    = Interpretor(self, "scale").eval
        self.interp_dimension   = Interpretor(self, "dimension").eval

    def iterunits(self):
        """ iterrator on unit string names """
        for u in self.unit_lookup:
            yield u     

    def iterkinds(self):
        """ iterator on kind string names """
        for k in self.kind_lookup:
            yield k         


def parentize(s):
    if any(o in s for o in "*/+-"):
        return "(%s)"%s
    return s

##################################################
#
# define here the key classes for kind computation  
#
###################################################
class Interpretor(object):
    def __init__(self, R, context):
        self.R = R        
        context_parser_lookup = {
            "kindhash": lambda r,i: _KindHash_(i[K_H]), 
            "unithash": lambda r,i: _KindHash_(i[U_H]),
            "unitlesshash": lambda r,i: _KindHash_(1.0 if r.kind_lookup[i[U_KIND]][K_ULESS] else i[U_H]),
            "unitkindname": lambda r,i: _KindName_(r, i[U_KIND]),
            "kindname": lambda r,i: _KindName_(r,i[K_NAME]),
            "kindbase": lambda r,i: _UnitName_(i[K_BASE]),
            "scale": lambda r,i: i[U_SCALE],
            "dimension": lambda r,i: _UnitDimension_({i[U_KIND]:i[U_DIMENSION]})
        }
        context_dict_lookup = {
           "kindhash": lambda r:r.kind_lookup, 
           "unithash": lambda r:r.unit_lookup,
           "unitlesshash": lambda r:r.unit_lookup,
           "unitkindname": lambda r:r.unit_lookup,
           "kindname": lambda r:r.kind_lookup,  
           "kindbase": lambda r:r.kind_lookup,
           "scale": lambda r:r.unit_lookup, 
           "dimension":lambda r:r.unit_lookup,  
        }
        context_return_lookup ={
           "kindhash": lambda o: getattr(o,"H", o),
           "unithash": lambda o: getattr(o,"H", o),
           "unitlesshash": lambda o: getattr(o,"H",o),
           "unitkindname": lambda o: getattr(o,"name",None),
           "kindname": lambda o: getattr(o,"name",None),
           "kindbase": lambda o: getattr(o,"name",None),
           "scale": lambda o:o, 
           "dimension": lambda o: dict((k,d) for k,d in getattr(o, "dimensions", {}).iteritems() if d)
        }
        self.d = context_dict_lookup[context](R)        
        self.f = context_parser_lookup[context]
        self.o = context_return_lookup[context]

    def __getitem__(self, item):
        return self.f(self.R, self.d[item])
    def __contains__(self, item):
        return item in self.d

    def keys(self):
        return self.d.keys()

    def eval(self, s):
        return self.o(eval(s, eval_globals, self))
        
class _KindHash_(object):
    def __init__(self, H):
        self.H = float(H)

    def __mul__(self, right):
        if isinstance(right, _KindHash_):
            return _KindHash_(self.H*right.H)    
        return self
    
    def __rmul__(self, left):
        if isinstance(left, _KindHash_):
            return _KindHash_(left.H*self.H)    
        return self

    def __div__(self, right):
        if isinstance(right, _KindHash_):
            return _KindHash_(self.H/right.H)    
        return self
    __truediv__ = __div__

    def __rdiv__(self, left):        
        if isinstance(left, _KindHash_):
            return _KindHash_(left.H/self.H) 
        return _KindHash_(1.0/self.H)

    __rtruediv__ = __rdiv__

    def __pow__(self, right):
        return _KindHash_(self.H**right)    


class _UnitDimension_(object):
    def __init__(self, dimensions):
        self.dimensions = dict(dimensions)
                
    def __mul__(self, right):

        if isinstance(right, _UnitDimension_):
            d = dict(self.dimensions)
            for kind,dim in right.dimensions.iteritems():
                if kind in self.dimensions:                    
                    d[kind] += dim
                else:                    
                    d[kind] = dim

            return _UnitDimension_(d)    
        return self
    
    def __rmul__(self, left):
        return self.__mul__(left)        
    
    def __div__(self, right):
        if isinstance(right, _UnitDimension_):
            d = dict(self.dimensions)
            for kind, dim in right.dimensions.iteritems():
                if kind  in self.dimensions:                    
                    d[kind] -= dim
                else:                    
                    d[kind] = -dim
            return _UnitDimension_(d)    
        return self

    __truediv__ = __div__

    def __rdiv__(self, left):
        d = dict(self.dimensions)
        for k,dim in d.iteritems():
            d[k] = -dim

        return left*_UnitDimension_(d)

    __rtruediv__ = __rdiv__

    def __pow__(self, right):
        d = dict(self.dimensions)
        for k,dim in d.iteritems():
            d[k] *= right
        return _UnitDimension_(d)



class _KindName_(object):
    def __init__(self, R, name):
        self.name = str(name)
        self.R = R

    def __mul__(self, right):
        if isinstance(right, _KindName_):
            if right.name==self.name:
                return _KindName_(self.R, kindofkind(self.R, "%s**2"%parentize(right.name)))
            return _KindName_(self.R, kindofkind(self.R, "%s*%s"%(self.name,right.name)))
        return self
    
    def __rmul__(self, left):
        if isinstance(left, _KindName_):
            if left.name==self.name:
                return _KindName_(self.R, kindofkind(self.R, "%s**2"%parentize(left.name)))
            return _KindName_(self.R, kindofkind(self.R, "%s*%s"%(left.name,self.name)))
        return self

    def __div__(self, right):
        if isinstance(right, _KindName_):
            return _KindName_(self.R, kindofkind(self.R, "%s/%s"%(parentize(self.name), parentize(right.name))))
        return self
    __truediv__ = __div__

    def __rdiv__(self, left):        
        if isinstance(left, _KindName_):
            return _KindName_(self.R, kindofkind(self.R, "%s/%s"%(parentize(left.name), parentize(self.name))))
        return _KindName_(self.R, kindofkind(self.R, "1/%s"%parentize(self.name)))

    __rtruediv__ = __rdiv__

    def __pow__(self, right):
        return _KindName_(self.R, kindofkind(self.R, "%s**%s"%(parentize(self.name),right)))

dummy = lambda x:x
class _UnitName_(object):
    def __init__(self, name):
        self.name = str(name)

    def __mul__(self, right):
        if isinstance(right, _UnitName_):
            if right.name==self.name:
                return _UnitName_(dummy("%s**2"%parentize(right.name)))
            return _UnitName_(dummy("%s*%s"%(self.name,right.name)))
        return self
    
    def __rmul__(self, left):
        if isinstance(left, _UnitName_):
            if left.name==self.name:
                return _UnitName_(dummy("%s**2"%parentize(left.name)))
            return _UnitName_(dummy("%s*%s"%(left.name,self.name)))
        return self

    def __div__(self, right):
        if isinstance(right, _UnitName_):
            return _UnitName_(dummy("%s/%s"%(parentize(self.name), parentize(right.name))))
        return self
    __truediv__ = __div__

    def __rdiv__(self, left):        
        if isinstance(left, _UnitName_):
            return _UnitName_(dummy("%s/%s"%(parentize(left.name), parentize(self.name))))
        return _UnitName_(dummy("1/%s"%parentize(self.name)))

    __rtruediv__ = __rdiv__

    def __pow__(self, right):
        return _UnitName_(dummy("%s**%s"%(parentize(self.name),right)))







##############################################################
#
#  unit the lookup tables
#
#############################################################




######################################################################################
#
#  Makers
#
######################################################################################


class UnitError(NameError):
    pass
def _extract_nameerror(e):
    try:
        return str(e).split("name ")[1].split(" is ")[0]
    except:
        return e    

class RegisterError(ValueError):
    pass    

M_PYTHON, M_SCALE, M_NAME, M_PRINT, M_LATEX = range(5)
K_NAME, K_PYTHON, K_DEFINITION, K_BASE, K_ULESS, K_H = range(6)
(U_UNIT, U_NAME, U_PRINT, U_DEFINITION, U_KIND, U_METRIX,
 U_SCALE, U_DIMENSION, U_ISMETRIX, U_H) = range(10)


def make_metrix(R, metrix, scale, name=None, prt=None, latex=None, callback=None):
    """ add a new metrik prefix for metrix units 

    Parameters
    ----------
    metrix : string,
        short prefix the metrix. (must be python compilant) 
    scale : float
        The scale of the metrix from the base unit, e.g. 'k' (kilo) is 1000.

    name : string, optional 
        long name for the metrix
    prt : string, optional
        A true string representation of metrix prefix. Used for nice print
    latex : string, optional
        LaTex representation of the metrix   
    
    """        
    metrix_info = [metrix, scale, 
                  name or metrix, prt or metrix, latex or metrix]
    R.metrix_lookup[metrix] = metrix_info
    for unit, info in R.unit_lookup.items():
        if info[U_ISMETRIX]:
            make_metrix_unit(R, info, metrix_info, callback)    

def remove_metrix(R, metrix, callback=None):
    if R.__isglobal__:
        raise RegisterError("Cannot remove metrix on global register")
    del R.metrix_lookup[metrix]
    removed = []
    for unit, info in R.unit_lookup.items():
        if info[U_METRIX]==metrix:
            removed.append(remove_unit(R, unit, callback=callback))
    return removed        

def make_kind(R, kind, definition="", baseunit="", name=None, unitless=False):
    """ Add a new kind of quantity

    Parameters
    ----------
    kind : string
        kind name (better if python compilant)

    definition : string
        If the kind derive from others give the mathematical string representation 
        of the new kind e.g. 
            >>> make_kind("velocity", "length/time")         
        If defintion is "" the new kind is completely independant.
    
    baseunit: string, optional
        The base unit of the kind (mostly the SI unit). e.g. is 'm' for 'length' kind

    name : string, optional
        A long name for the new kind.    
    

    """    
    if kind in R.kind_lookup:
        raise ValueError("kind '%s' already exists"%kind)

    name = name or kind

    if definition:
        H = R.interp_kindhash(definition)
    else:
        H = hash(kind)

    if H in R.hash2kind_lookup:
        print("Warning '%s' kind shares the same hashing than '%s'."%(kind,R.hash2kind_lookup[H]))
            

    R.hash2kind_lookup[H] = kind        
    R.kind_lookup[kind] = [name, kind, definition, baseunit, unitless, H]
    R.scale2unit_lookup[kind] = {}

    
        


def remove_kind(R, kind, callback=None):
    if R.__isglobal__:
        raise RegisterError("Cannot remove kind on global register")    

    H = R.kind_lookup[kind][K_H]    
    del R.hash2kind_lookup[H]
    del H
        
    del R.kind_lookup[kind]
    
    

    removed = []
    for unit, info in R.unit_lookup.items():
        if info[U_KIND]==kind:
            removed.append(remove_unit(R, unit, callback=callback))

    del R.scale2unit_lookup[kind]

    return removed        
                    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Units

def make_unit(R, unit, scale_or_definition, kind, dimension=1, metrix=False, name=None, prt=None, callback=None):
    """ Make a new unit

    Parameter
    ---------
    
    unit : string
        unit short name (better if python compilant)
        
    scale : float or string
        the unit scale. For instance for 'deg' : 180/pi.  

    kind : string
        the kind name of the unit. e.g. 'length' for 'm'. The kind must have been previously created
    
    dimension : int, optional
        Unit dimension used for metrix system units. Default is 1
        for instance a volume is 3, area is 2.
        If metrix is False, this has no effect. 

    metrix : bool, optional
        if True create also all the units derived from this unit. 
        e.g. make_unit("length", "m", 1.0, metrix=True) will also create the 
            "km", "dm", "cm", "mm", etc ....

    name : string, optional
        long name for the unit 

    prt : string, optional
        short representation for nice printout purpose


    """    
                
    #except:
    #    dimension = 1 

    if unit in R.unit_lookup and R.__isglobal__:
        raise ValueError("Unit '%s' already exists in the global unit register. You need to create your hown register"%unit)

    definition = scale_or_definition
    if isinstance(scale_or_definition, basestring):
        scale = R.interp_scale(scale_or_definition)        
    else:
        scale = float(scale_or_definition)
        
    H = R.kind_lookup[kind][K_H]    


    #if dimension>1:        
    #    scale = scale**dimension
        
    uinfo = [unit, # U_UNIT  python unitname 
             name, # U_NAME  nice name 
             prt,  # U_PRINT print name 
             definition, # U_DEFINITION initial definition for reference
             kind, # U_KIND unit kind 
             "",  # U_METRIX   metrix suffix of the unit 
             scale, # U_SCALE  scale of the unit 
             int(dimension),  # U_DIMENSION  unit dimension e.g. 2 for area
             bool(metrix),     # U_ISMETRIX
             H # U_H
            ]
                
    R.unit_lookup[unit] = uinfo
    R.scale2unit_lookup[kind].setdefault(scale, unit)


    
    if callback:      
        callback(unit)

    if metrix:
        for metrix_info in R.metrix_lookup.itervalues():
            make_metrix_unit(R, uinfo, metrix_info, callback)

def remove_unit(R, unit, callback=None):
    if R.__isglobal__:
        raise RegisterError("cannot remove unit on global register")
    kind = R.unit_lookup[unit][U_KIND]
    scale = R.unit_lookup[unit][U_SCALE]

    del R.unit_lookup[unit]
    del R.scale2unit_lookup[kind][scale]


    if callback:      
        callback(unit)
    return unit
        

def make_metrix_unit(R, unit, metrix, callback=None):
    metrix_info = R.metrix_lookup[metrix] if isinstance(metrix, basestring) else metrix
    unit_info   = R.unit_lookup[unit] if isinstance(unit, basestring) else unit

    metrix = metrix_info[M_PYTHON]
    unit = unit_info[U_UNIT]
    metrix_unit = metrix+unit
    metrix_scale = metrix_info[M_SCALE]
    name = unit_info[U_NAME]
    prt = unit_info[U_PRINT]
    dimension = unit_info[U_DIMENSION]
    kind = unit_info[U_KIND]


    if dimension>1:
        metrix_scale = metrix_scale**dimension

    metrix_scale *= unit_info[U_SCALE] # multiply by the unit scale

    newinfo = [metrix_unit, 
               metrix_info[M_NAME]+" "+(name if name else unit), 
               metrix_info[M_PRINT]+(prt if prt else unit), 
               unit_info[U_DEFINITION], 
               kind,
               metrix, 
               metrix_scale,
               dimension,
               False, 
               R.kind_lookup[kind][K_H]
            ]
            
    R.unit_lookup[metrix_unit] = newinfo
    R.scale2unit_lookup[kind].setdefault(metrix_scale, metrix_unit)


    if callback:
        callback(metrix_unit)        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Convertos



def make_convertor(R, kinds, targets, func):
    """ make a kind to kind convertor 

    This is rarely used, most of the unit of the same kind are just proportional from 
    one to the other : foot, meter, km ....
    A counter example is the temperature between Kelvin and Celcius, one need to create
    two kinds 'temperature' and 'temperature_c' and make a link between them.

    Parameters
    ----------
    kinds : string or list
        list of kinds. If string must be separated by ","
    targets: string or list
        list of kinds targeted by the convertor

    func : callable
        the conversion function from kind in kinds to kind in targets             

    Example
    -------
        >>> make_kind('temp')
        >>> make_kind('temp_c')    
        >>> make_convertor('temp', 'temp_c', lambda tk,u:  convert(tk,u,'K') - 273.15   )
        >>> make_convertor('temp_c','temp' , convert(tc,u,'Cel') + 273.15 )
    """
    convertor_lookup = R.convertor_lookup
    kind_lookup = R.kind_lookup

    if kinds == "*":
        if targets == "*":
            raise ValueError("building convertor kind and target cannot be both '*'")

    if isinstance(kinds, basestring):
        kinds =  [k.strip() for k in kinds.split(",")]
    
    if isinstance(targets, basestring):       
        targets  = [t.strip() for t in targets.split(",")]
    
    if isinstance(func, basestring):
        convertor_func = eval(func)
    else:
        convertor_func = func    

    for kind in kinds:
        for target in targets:
            if kind==target:
                raise ValueError("building convertor kinds are equal '%s'"%kind)
            convertor_lookup[(kind, target)] = convertor_func   
    return

    if kinds == "*":
        if targets == "*":
            raise ValueError("building convertor kind and target cannot be both '*'")
        kinds = kind_lookup.keys()

    elif isinstance(kinds, basestring):
        kinds =  [k.strip() for k in kinds.split(",")]
    if targets == "*":
        ## all the other kind are targeted
        ## should be used only to dimless kinds  
        targets = kind_lookup.keys()
        for kind in kinds:
            targets.remove(kind)
    elif isinstance(targets, basestring):       
        targets  = [t.strip() for t in targets.split(",")]
    
    if isinstance(func, basestring):
        convertor_func = eval(func)
    else:
        convertor_func = func

    for kind in kinds:
        for target in targets:
            convertor_lookup[(kind, target)] = convertor_func   

    ######################################################################################
    #
    #  Define the API function.
    #
    ######################################################################################

def convert(R, value, unit, newunit, inside=lambda x,u:x):
    if hasattr(value, "__tovalue__"):
        value = value.__tovalue__(value)        
             
    kind = kindofunit(R, unit)
    newkind = kindofunit(R, newunit)

    scale    = scaleofunit(R, unit)
    newscale = scaleofunit(R, newunit)

    if kind != newkind:
        convertor = getconvertor(R, kind, newkind)
        if convertor is None:
            raise ValueError("cannot convert a '%s' to a '%s'"%(kind, newkind))
        newvalue, convertedunit = convertor(value, unit)
        if convertedunit!=newunit:
            newvalue = convert(R,newvalue, convertedunit, newunit)
    else:
        newvalue = value / newscale * scale    
    return inside(newvalue, newunit)
        

def _kindofunit(R, unit):
    """ return the kind string of a unit or None if unit does not exist"""        
          
    try:
        H = R.interp_unithash(unit)
    except NameError as e:
        raise UnitError("unknown unit %s in '%s'"%(_extract_nameerror(e), unit))        
    return R.hash2kind_lookup.get(H,None)

def kindofunit(R, unit):
    """ return the kind string of a unit or None if unit does not exist"""
    k = _kindofunit(R, unit)
    if k is not None:
        return k

    try:
        H = R.interp_unitlesshash(unit)
    except NameError as e:
        raise UnitError("unknown unit %s in '%s'"%(_extract_nameerror(e), unit))
        
    try:
        kind = R.hash2kind_lookup[H]
    except KeyError:
        return anykindofunit(R,unit)
    return kind 

def anykindofunit(R, unit):    
    try:
        kind = R.interp_unitkindname(unit)
    except NameError as e:
        raise UnitError("unknown unit %s : %s"%(unit,_extract_nameerror(e)))
    return kind

def scaleofunit(R, unit):
    """ return the scale factor of a unit """
    try:
        scale = R.interp_scale(unit) if unit else 1.0
    except NameError as e:
        raise UnitError("unknown unit %s : %s"%(unit,_extract_nameerror(e)))        
    return scale

def kindofkind(R, kind):
    try:
        H = R.interp_kindhash(kind)
    except NameError as e:
        raise UnitError("unknown kind : %s"%_extract_nameerror(e))
    try:
        newkind = R.hash2kind_lookup[H]
    except KeyError:
        return kind
    else:
        return newkind                

def unitofunit(R, unit,kind=None):
    """ rewrite a unit operation in a intelligible unit string if possible 

    Parameters
    ----------
    unit : string
        unit expression 

    Returns
    -------
    unit : string
        reformed string
    kind : string
        the kind of quantity    

    Example
    -------
    >>> unitofunit( "m*m*m")
    ("m3", "volume")
    >>> unitofunit( "3600*s")
    ("h", "time")
    """
    kind = kindofunit(R, unit) if kind is None else kind
    if any(o in unit for o in '*/+-'):
        scale = scaleofunit(R, unit)
        return unitofscale(R, scale, kind) or unit, kind
    return unit, kind

def unitofscale(R, scale, kind):
    """ return the string unit given a scale and a kind 

    Parameters
    ----------
    scale : float or None
        the scale factor of the researched unit or None if failure
    kind : string
        the quantity kind

    Outputs
    -------
    unit : string or None
        the found unit or None if unit does not exists

    Example
    -------
    >>> unitofscale( 1.0, "length")
    "m"
    >>> unitofscale( 1000, "length")
    "km"
    >>> unitofscale( 3600*24, "time")
    "d"

    """
    return R.scale2unit_lookup.get(kind,{}).get(scale, None)    

def hashofkind(R, kind):
    return R.kind_lookup[kind][K_H]

def hashofunit(R, unit):
    try:
        H = R.interp_unithash(unit)
    except NameError as e:
        raise UnitError("unknown kind : %s"%_extract_nameerror(e))    
    return H

def _getkindinfo(R, kind):
    return R.kind_lookup.get(kind,None)

def _getunitinfo(R, unit):
    unit, kind = unitofunit(R, unit)
    return R.unit_lookup.get(unit, None)

def unitof(value):
    """ return unit of a value 

    return getattr(value, "unit", "") 
    
    Parameter
    ---------
    value : numerical like, quantity

    Outputs
    -------
    unit : string
    """
    return getattr(value, "unit", "")    

def kindof(value):
    """ return kind of a value 

    return getattr(value, "kind", "") 
    
    Parameter
    ---------
    value : numerical like, quantity

    Outputs
    -------
    unit : string
    """
    return getattr(value, "kind", None)

def valueof(value):
    """ return value of a quantity or value itself

    return getattr(value, "_value", value) 
    
    Parameter
    ---------
    value : numerical like, quantity

    Outputs
    -------
    value : numerical
    """
    return getattr(value, "_value", value)

def scaleof(R, value):
    """ return the 'scale' of a quantity or value itself

    return scaleofunit( getattr(value, "unit", '') ) 

    Parameter
    ---------
    value : numerical like, quantity

    Outputs
    -------
    value : numerical
    """
    return scaleofunit(R, getattr(value, "unit", ''))    

def isunitless(value):
    """ return True is value is unitless

    Parameter
    ---------
    value : numerical like, quantity

    Outputs
    -------
    test : bool
    """
    kind = kindof(value)
    R = getattr(value, "R", None)
    if R:    
        return (kind is None and unitof(value)=='') or\
               (hashofkind(R,kind)==1) or\
               (R.kind_lookup[kind][K_ULESS])
    else:
        return (kind is None and unitof(value)=='')
               




def unitsofkind(R, kind):
    """return the list of units of a quantity kind 

    Parameters
    ----------
    kind : string
        quantity kind

    Outputs
    -------
    units : list of string 
        list of string units of that `kind`
                    
    """
    return [k for k,i in R.unit_lookup.iteritems() if i[U_KIND]==kind]

def printofunit(R, unit):
    """ return the 'print' string version of a unit

    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    print_version : string
        if not print version is found, unit is returned

    Examples
    --------
    >>> printofunit("mm")
    u'millimeter' 
    """
    info = _getunitinfo(R, unit)
    return unicode(info[U_PRINT]) if info else unit

def definitionofunit(R, unit):
    """ return the  string definition of a unit or None
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    definition : string or None
        python executable string 

    Examples
    --------
    >>> definitionofunit( "_c" )  # speed of light 
    '299792458*m/s'

    """
    info = _getunitinfo(R, unit)
    return info[U_DEFINITION] if info else None

def nameofunit(R, unit):
    """ return the  string name of a unit or None
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    name : string or None
    
    Examples
    --------
    >>> nameofunit( "_c" )     
    u'velocity of light'
    """
    info = _getunitinfo(R, unit)
    return unicode(info[U_NAME]) if info else None

def metrixofunit(R, unit):
    """ return the  string metrix of a unit or None
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    name : string or None
    
    Examples
    --------
    >>> metrixofunit( "km" )     
    'k'
    """
    info = _getunitinfo(R, unit)
    return unicode(info[U_METRIX]) if info else None

def baseofunit(R, unit):
    """ return the  string base of a unit or None
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    name : string or None
    
    Examples
    --------
    >>> baseofunit( "km" )     
    'm'
    """
    return baseofkind(R, kindofunit(R, unit))
    #info = _getunitinfo(unit)
    #return unicode(info[U_BASE]) if info else None

def basescaleofunit(R, unit):
    """ return the  scale base of a unit or 1.0
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    scale : float
    
    Examples
    --------
    >>> basescaleofunit( "km" )     
    1000.0
    >>> getscale( "m" ) 
    1.0
    """    
    return scaleofunit(R, unit)/scaleofunit(R, baseofunit(R, unit))
            
def dimensionsofunit(R, unit):
    """ return the dimentions of a unit

    Result is return in a dictionary where keys are kind name 
    and values are dimension 
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    dimensions : dictionary
    
    Examples
    --------
    >>> dimensionsofunit("km/s**2 * g")
    {'length': 1, 'mass': 1, 'time': -2}
    """
    try:
        dims = R.interp_dimension(unit)
    except NameError as e:
        raise UnitError("unknown kind : %s"%_extract_nameerror(e))    
    return dims                  


def baseofkind(R, kind):
    """ return the 'base' unit of a quantity kind 

    The base unit is from the SI. 

    Parameters
    ----------
    kind : string
        kind quantity 

    Outputs
    -------
    units :  string or None
        base unit of that `kind` or None if kind not found 

    Examples
    --------
    >>> baseofkind("velocity")
    'm/s'
    """
    kind = kindofkind(R, kind)
    try:
        kindinfo = R.kind_lookup[kind]
    except KeyError:
        try:
            unit = R.interp_kindbase(kind)
        except NameError as e:
            #return None
            raise UnitError("Kind unknown : %s"%_extract_nameerror(e))            
        return unitofunit(R,unit)[0]

    if kindinfo:
        return kindinfo[K_BASE]  
    return None

def isunitof(R, unit, kind):
    """ return True if unit is of kind 

    Parameters
    ----------
    unit : string

    kind : string

    Outputs
    -------
    test : boolean 

    Examples
    --------
    >>> isunitof( "m", "length")
    True
    >>> isunitof( "Hz", "length")
    False 
    """
    return kindofunit(R, unit) == kind

def getconvertor(R, kind, kind_targeted):
    """ return a convertor funtion between 2 kinds if exist or None

    For most of the kinds, units are just a scale factor between eachother.
    However some unit transformation cannot be donne with a scale factor so convertor
    is used as a bridge between two kinds.
    A good example is the temperature, they are to kinds for temperature : 
        'temperature' (Kelvin based) 'temperature_c' (Celsius base)

    User normaly do not need this function, the conversion is transparent.
    e.g. :   quantity(300, "k").to("Cel") ->  26.85 [Cel]

    """
    try:
        c = R.convertor_lookup[(kind, kind_targeted)]
    except KeyError:
        try:
            c = R.convertor_lookup[("*", kind_targeted)]
        except KeyError:
            try:
                c = R.convertor_lookup[(kind, "*")]
            except:
                c = None
    return c

def linksofkind(R, kind):
    """ return a list of kinds for which `kind` is linked 

    Parameters
    ----------
    kind : string
        quantity kind

    Outputs
    -------
    lst : list of string
        linked kind list 

    Examples
    --------
    >>> linksofkind('temperature')
    ['temperature_c', 'fraction']            
    """
    lst = []
    for ckind, target in R.convertor_lookup.keys():
        if kind == ckind:
            lst.append(target)
    return lst

def arekindconvertible(R, kind_from, kind_to):
    if hashofkind(R, kind_from)==hashofkind(R, kind_to):
        return True
    return kind_to in linksofkind(R, kind_from)

def areunitconvertible(R, unit_from, unit_to):
    kind_from = kindofunit(R, unit_from)
    kind_to = kindofunit(R, unit_to)
    if (kind_to is None) or (kind_from is None): 
        return False
    return arekindconvertible(R, kind_from, kind_to)

def getkinds(R):
    """ return a list of existing kind 

    Output
    ------
    kinds : list of string  
        list of kind
    """
    return R.kind_lookup.keys()

def getunits(R):
    """ return a list of existing units 

    Output
    ------
    units : list of string  
        list of unit

    See Also
    --------
    unitsofkind  : return a list of unit for a given kind only
    """
    return R.unit_lookup.keys()

def unitexist(R, unit):
    """ True is unit exist in the register

    Parameters
    ----------
    unit : string
                
    Output
    ------
    test : boolean 
    """
    return unit in R.unit_lookup

def kindexist(R, kind):
    """ True is kind exists in the register

    Parameters
    ----------
    kind : string
        quantity kind

    Output
    ------
    test : boolean 
    """
    return kind in R.kind_lookup

def metrixexist(R, metrix):
    """ True is metrik exist in the register

    Parameters
    ----------
    metrix : string
                
    Output
    ------
    test : boolean 
    """
    return metrix in R.metrix_lookup
    
