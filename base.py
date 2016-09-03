from .api2 import (kindofunit, unitofunit, scaleofunit, 
                  basescaleofunit,
                  printofunit,                                    
                  unitof, kindof, isunitless, valueof, 
                  scaleof, baseofunit, arekindconvertible, 
                  nameofunit, convert, 
                  UnitError, 
                  Registery
                )
from .registery import QuantityTypes
from . import units
from . import kinds

__all__ = ["Qfloat", "Qint", "quantity", "Qany", "Unit"]

Rglobal = Registery()
Rglobal.__isglobal__ = True
QTglobal = QuantityTypes(Rglobal)



#########################################################


def getunit(o):
    """ return unit of object or None"""
    #if not hasattr(o, "_value"):
    #   return None
    return getattr(o, "unit", None)

def getvalue(o):
    return getattr(o, "_value", o)

def getkind(o):
    """ return kind of object or None"""
    if not hasattr(o, "_value"):
        return None    
    return getattr(o, "kind", None)

def same_kind_operation_setting(error, mode="drop"):
    """ Setup the behavior of addition or substraction of quantities wihout the same unit 

    Seutp the behavior of a+b when a and b are two quantities with the same *kind* (e.g. 'length')
    but not the same scale unit (e.g. 'm' vs 'cm'). This behavior is set with two parameter explain 
    below. 
    Let considers below as an example
        >>> a = quantity(1.0, "m") 
        >>> b = quantity(1.0, "cm")

    Parameters
    ----------
    error : string
        - if "error" : raise a RunTimeError exception when a+b occurs 
        - if "warning" : print a warning when this occurs
        - if "silent" :  print or raise nothing
    
    mode : string
        - if "drop" :  a+b return  2.0 the unit is simply dropped. 
        - if "keep left" : a + b = 1.01 [m]  ;  b + a = 101 [cm]
            keep the unit of the left operand
        - if "keep right" : a + b = 101 [cm] ;  b + a = 1.01 [m]
            keep the unit of the right operand
        - if "keep highest" :  a + b = 1.01 [m] ;  b + a = 1.01 [m]   
            keep the unit with the highest scale
        - if "keep lowest" :  a + b = 101 [cm] ;  b + a = 101 [cm]   
            keep the unit with the lowest scale             
    
        mode has no effect if error=="error"

    """ 
    global same_kind, same_kind_msg

    error_lookup = ["silent", "warning", "error"]
    mode_lookup  = ["keep left", "keep right", "keep highest", "keep lowest", "drop"]
    if error not in error_lookup:
        raise ValueError("error should be one of '%s' got '%s'"%("', '".join(error_lookup), error)) 
    if mode not in mode_lookup:
        raise ValueError("mode should be one of '%s' got '%s'"%("', '".join(mode_lookup), mode)) 

    if error == "error":
        def same_kind_msg(left, right):        
            raise RuntimeError("Adding or substracting '%s' with '%s'"%(unitof(left), unitof(right)))

    elif error == "warning":
        def same_kind_msg(left, right):
            print("Warning: Adding or substracting '%s' with '%s'"%(unitof(left), unitof(right)))
    else:
        def same_kind_msg(left, right):
            pass        

    if mode == "keep right":        
        def same_kind(left, right):
            same_kind_msg(left, right)
            unit = unitof(right)
            if error is "warning":
                print("  unit will be '%s'"%unit)
            return left.to(unitof(right))._value, right._value, unit

    elif mode == "keep left":
        def same_kind(left, right):
            same_kind_msg(left, right)
            unit = unitof(left)
            if error is "warning":
                print("  unit will be '%s'"%unit)
            return left._value, right.to(unitof(left))._value, unit    

    elif mode == "keep lowest":
        def same_kind(left, right):
            same_kind_msg(left, right)
            lscale, rscale = scaleofunit(left.R, unitof(left)), scaleofunit(right.R, unitof(right))
            if lscale>rscale:
                unit = unitof(right)
                if error is "warning":
                    print("  unit will be '%s'"%unitof(right))
                return left.to(unitof(right))._value, right._value, unit
            else:
                unit = unitof(left)                
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left._value, right.to(unitof(left))._value, unit

    elif mode == "keep highest":
        def same_kind(left, right):
            same_kind_msg(left, right)
            lscale, rscale = scaleofunit(left.R, unitof(left)), scaleofunit(right.R, unitof(right))
            if lscale<rscale:
                unit = unitof(right)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left.to(unitof(right))._value, right._value, unit
            else:
                unit = unitof(left)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left._value, right.to(unitof(left))._value, unit

    elif mode == "keep base":
        def same_kind(left, right):
            same_kind_msg(left, right)
            lscale, rscale = scaleofunit(left.R, unitof(left)), scaleofunit(right.R, unitof(right))
            if abs(lscale-1)>abs(rscale-1):
                unit = unitof(right)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left.to(unitof(right))._value, right._value, unit
            else:
                unit = unitof(left)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left._value, right.to(unitof(left))._value, unit    


    elif mode == "drop":            
        def same_kind(left, right):            
            same_kind_msg(left, right)
            if error is "warning":
                    print("  unit is dropped")
            return left._value, right._value, ''
    else:
        raise ValueError("mode '%s' invalid"%mode)        
####

same_kind_operation_setting("silent", "keep highest")

def different_kind_operation_setting(error):
    global different_kind
    error_lookup = ["silent", "warning", "error"]
    if error not in error_lookup:
        raise ValueError("error should be one of '%s' got '%s'"%("', '".join(error_lookup), error))     

    if error == "error":                                 
        def different_kind(left, right):
            raise RuntimeError("Adding or substracting '%s' with '%s'"%(kindof(left)or'unitless', kindof(right)or'unitless'))
            

    elif error == "warning":                   
        def different_kind(left, right):
            print("Warning: Adding or substracting '%s' with '%s'"%(kindof(left)or'unitless', kindof(right)or'unitless'))
            return valueof(left), valueof(right), ''
    else:
        def different_kind(left, right):
            return valueof(left), valueof(right), ''

different_kind_operation_setting("error")           

#########################################################

def unitless_operation_setting(error, mode="drop"):
    global unitless_kind, unitless_kind_msg
    error_lookup = ["silent", "warning", "error"]
    mode_lookup = ["drop", "keep"]

    if error not in error_lookup:
        raise ValueError("error should be one of '%s' got '%s'"%("', '".join(error_lookup), error))     

    if mode not in mode_lookup:
        raise ValueError("mode should be one of '%s' got '%s'"%("', '".join(mode_lookup), mode))     
    

    if error == "error":
        def unitless_kind_msg(left, right):
            raise RuntimeError("Adding or substracting %s with  %s"%(kindof(left)or'unitless', kindof(right)or'unitless'))
    elif error == "warning":
        def unitless_kind_msg(left, right):
            print("Warning: Adding or substracting %s with  %s"%(kindof(left)or'unitless', kindof(right)or'unitless'))
    
    if mode is "keep":
        def unitless_kind(left, right):
                unitless_kind_msg(left, right)            
                return valueof(left), valueof(right), left.unit if hasattr(left, "unit") else right.unit
    else:
        def unitless_kind(left, right):
            unitless_kind_msg(left, right)
            return valueof(left), valueof(right), ''

unitless_operation_setting("warning", "drop")            

##########################################################
def _linear_op_prepare(left, right):
    """ prepare the quantity for an addition or substraction 

    take the left and right quantity for the operation : left+right or lef-right
    return lv, rv, unit   :  left value , right value and the new unit that gonna be used
    """   
    if isinstance(right, Unit):
        right = right.new(1.0)
    if isinstance(left, Unit):
        left = left.new(1.0) 
             
    if isunitless(left):
        if isunitless(right):
            ## two unit less values. safe to add
            lv, rv, unit = valueof(left), valueof(right), ''
        else:
            ## One is unitless, check is it is allowed 
            lv, rv, unit = unitless_kind(left, right)          
    else:       
        if isunitless(right):
            ## One is unitless, check is it is allowed 
            lv, rv, unit = unitless_kind(left, right)
        else:
            lkind, rkind = kindof(left), kindof(right)
            if not arekindconvertible(left.R, lkind, rkind):
                ## Two values has different kind 
                lv, rv, unit = different_kind(left, right)

            elif scaleof(left.R, left)!=scaleof(right.R, right) or (lkind!=rkind):
                ## The two values has different units 
                ## note that lkind can be != than rkind if they are linked by some func                
                lv, rv, unit = same_kind(left, right)            
            else:
                ## safe for operation they have both same kind and same unit                
                lv, rv, unit = valueof(left), valueof(right), unitof(left)
    return lv, rv, unit

def _compare_op_prepare(left, right):
    """ prepare the quantity for a comapraison  

    take the left and right quantity for the operation : left<right or lef>right, etc..
    return lv, rv :  left value , right value
    """ 
    if isinstance(right, Unit):
        right = right.new(1.0)
    if isinstance(left, Unit):
        left = left.new(1.0)        

    if isunitless(left):
        # safe to compare 
        return valueof(left), valueof(right)              
    else:       
        if isunitless(right):
            ## safe to compare
            return valueof(left), valueof(right) 
        else:
            lv = getvalue(left)
            try:
                rv = right.to(left.unit)
            except ValueError:
                raise TypeError("cannot compare a '%s' to a '%s' "%(kindof(left), kindof(right)))            
    return lv, rv




#########################################################

def quantity(R, value=0.0, unit='', QT=QTglobal):    
    # for cl, _, quantityclass in value2type_lookup:
    #     if isinstance(value, cl):            
    #         break
    
    # else:
    #     for _, parser, quantityclass in value2type_lookup:
    #         try:
    #             parser(value)
    #         except TypeError:
    #             continue
    #         else:
    #             break
    #     else:
    #         raise TypeError("incopatible type '%s' for quantity"%type(value))
    if not unit:
        return value
    
    ##
    # To avoid embigous stuff like 1*km+1*m -> 1 km
    # if user wants a int it shoudl explicitely ask for it
    #    
    if isinstance(value, int):
        value = float(value)    
        
    quantityclass, value = QT.get_quantity_class(value)
    if quantityclass is None:
        raise TypeError("incompatible type '%s' for quantity"%type(value))
          
    if isinstance(unit, basestring):
        unit = parsetrueunit(R, unit)
    else:
        unit = [parsetrueunit(R, u) for u in unit]
    
    if issubclass(quantityclass,Unit):
        return quantityclass(unit)          
    return quantityclass(value, unit)

def clone(value, unit, QT=QTglobal):
    """ clone a value to a unitary quantity object with the given unit 
    and the class of value
    """
    cl, _ = QT.get_quantity_class(getvalue(value))    
    if cl is None:
        raise ValueError("Cannot make a quantity with value of type '%s'"%type(value))

    if hasattr(cl, "unitary_quantity"):
        return cl.unitary_quantity(unit)
    else:        
        return cl(1.0, unit)    

def parseunit(R, unit):
    return unit
    try:
        u, _ = unitofunit(R, unit)
    except UnitError:
        u = str(unit)        
    return u

def parsetrueunit(R, unit):
    try:
        u, _ = unitofunit(R, unit)
    except UnitError:
        u = str(unit)        
    return u

def parentize(unit):
    if any(o in unit for o in "*/+-"):
        return "(%s)"%unit
    return unit

######################################################################
#
# We define here the base quantity functions separatly because it appears
# that subclassing in numpy.floatxx or numpy.intxx makes segmentation fault 11
# 
######################################################################

_shared_funcs = {}
def new(self, value):
    """ build a new quantity of same unit with new value """
    return self.__class__(value, self.unit)  
_shared_funcs["new"] = new
del new


def unit(self):
    """ quantity unit """
    return self._unit
_shared_funcs["unit"] = property(unit)
del unit

def _value(self):
    """ float representation of quantity """
    return self.__tovalue__(self)
_shared_funcs["_value"] = property(_value)
del _value

def unitname(self):
    return nameofunit(self.R, self.unit)
_shared_funcs["unitname"] = property(unitname)
del unitname    

def kind(self):
    """ quantity kind value """
    return kindofunit(self.R, self._unit)
    #return self._kind
_shared_funcs["kind"] = property(kind)
del kind  

def unitscale(self):
    """ quantity scale compare to the base unit of its kind """
    return basescaleofunit(self.R, self._unit)    
_shared_funcs["unitscale"] = property(unitscale)
del unitscale  

def unitdimensions(self):
    """ dimensions of unit """
    return unitdimensions(self.R, self._unit)    
_shared_funcs["unitdimensions"] = property(unitdimensions)
del unitdimensions  


def tobase(self):
    """ quantity transform to its kind base unit """    
    return self.to(self.baseunit)
_shared_funcs["tobase"] = tobase
del tobase  


def unitary_quantity(cl, unit):
    """ quantity scale value """
    return cl(1.0, unit)
_shared_funcs["unitary_quantity"] = classmethod(unitary_quantity)
del unitary_quantity

def baseunit(self):
    """ The string unit coresponding to the base of its kind """
    return baseofunit(self.R, self.unit)
_shared_funcs["baseunit"] = property(baseunit)
del baseunit

def __init_unit__(self):
    self.QT = self.QT()
_shared_funcs["__init_unit__"] = __init_unit__
del __init_unit__


def to(self, newunit):
    """ convert the quantity to a new unit e.g 'm' to 'km'"""
    if not self._unit :
        raise ValueError('quantity has no unit')
    
    if not isinstance(newunit, basestring):
        newunit = getunit(newunit)
        if newunit is None:
            raise ValueError("new unit must be a string or must have the 'unit' atribute")

    ## save sometime return self if the unit is unchanged
    ## should not be a problem because self is un-mutable
    if newunit == self._unit:
        return self
    return convert(self.R, self.__tovalue__(self), self.unit, newunit, 
                   self.__class__ if not hasattr(self, "__qwrapper__") else self.__qwrapper__)

#del to
#####
# test 

class _UnitConverterProperty(object):
    def __init__(self, unit):
        self.unit = unit
        self.__doc__ = "value converted to %s"%unit

    def __get__(self, obj, cl=None):
        if obj is None:
            if cl:
                return cl.__qbuilder__(1.0, self.unit)
            return self    
        return obj.to(self.unit)        


class UnitConverterProperty(object):
    def __init__(self, unit):
        self.unit = unit
        self.__doc__ = "value converted to %s"%unit

    def __get__(self, convertor, cl=None):
        if convertor is None:
            if cl:
                if hasattr(cl, "unitary_quantity"):
                    return cl.unitary_quantity(self.unit)
                else:                    
                    return cl(1.0, self.unit)
            return self
        return convertor(self.unit)        
        #return obj.to(self.unit)     


class UnitsCollection:
    """ just a collection of attribute """
    pass

class KindsCollection:
    pass


class ConvertorInstance(object):
    def __init__(self, convertor, cl, obj):
        self.convertor = convertor
        self.cl  = cl
        self.obj = obj        
            
    def __getitem__(self, item):
        if isinstance(item, tuple):
            return (self(i) for i in item)
        return self(item)
                        
    def __call__(self, *args):

        if len(args)>2:
            raise ValueError("to takes zero to two argument")
        if not args:
            unit = None
        elif len(args)==2:
            if issubclass(self.cl, Unit):
                v,u = args
                return quantity(self.cl.R, v, u, self.cl.QT())
            else:    
                return self.cl(*args)        
        else:
            unit, = args        

                
        if self.obj is None:
            if unit is None:
                raise ValueError("provide a unit")
            elif not isinstance(unit, basestring):
                unit = getunit(unit)
                if unit is None:
                    raise ValueError("new unit must be a string or must have the 'unit' atribute")

            unit, _ = unitofunit(self.cl.R, unit or '')
            if hasattr(self.cl, "unitary_quantity"):
                return self.cl.unitary_quantity(unit)
            else:    
                return self.cl(1.0, unit)

        if unit is None:
            unit = self.obj.unit
        elif not isinstance(unit, basestring):
            unit = getunit(unit)
            if unit is None:
                raise ValueError("new unit must be a string or must have the 'unit' atribute")
                
        unit,_ = unitofunit(self.obj.R, unit)
        if isinstance(self.obj, Unit):
            return  to(self.obj.new(1.0), unit)
        return to(self.obj, unit)
    
    def new(self, value, unit=None):
        if unit is None:
            if self.obj is None:
                return self.cl(value)
            unit, _ = unitofunit(self.obj.R, self.obj.unit) 
            return self.cl(value, unit)
                    
        unit, _ = unitofunit(self.obj.R, unit)            
        return self.cl(value, unit)

    @property
    def base(self):
        if self.obj is None:
            raise Exception("'to' called out of object context")
        return self.__call__(baseofunit(self.obj.R, self.obj.unit))
        
    def __getattr__(self, attr):
        return self.__call__(attr)        

class BaseConvertorInstance(ConvertorInstance):
    pass

class Convertor(object):
    Instance = BaseConvertorInstance    
    def __get__(self, obj, cl=None):
        if obj is None:
            if cl:
                return self.Instance(self, cl, None)
            return self    
        return self.Instance(self, type(obj), obj)

################################
_shared_funcs["to"] = Convertor()
_shared_funcs["R"] = Rglobal

_shared_funcs["_QTwr_"] = staticmethod(lambda :None)
def QT(cl):
    return cl._QTwr_() or QTglobal  
_shared_funcs["QT"] = classmethod(QT)
del QT

def __mul__(left, right):
    runit = getunit(right)
    lunit = left.unit
    lval = left._value
    #if isinstance(right, tuple(left._ref_classes)):
    if runit is not None:
        try:
            rval = right._value
        except:
            # this happen for units without value
            return NotImplemented
                                        
        if left.unit:
            v, unit = lval*rval, "%s*%s"%(lunit, right.unit)
        else:
            v, unit = lval*rval, "%s*%s"%(lval, right.unit)    
    else:
        if lunit:
            v, unit = lval*right, "%s"%(lunit)
        else:
            v, unit = lval*right, ''
    return left.__qbuilder__(v, unit)
_shared_funcs["__mul__"] = __mul__
del __mul__

def __rmul__(right, left):       
    #if isinstance(left, tuple(right._ref_classes)):
    lunit = getunit(left)
    runit = right.unit
    rval  = right._value 
    if lunit is not None:
        try:
            lval = left._value
        except:
            # this happen for units without value
            return NotImplemented

        if runit:
            v, unit = lval*rval, "%s*%s"%(lunit, runit)
        else:
            v, unit = lval*rval, "%s*%s"%(lval, runit)    
    else:
        if runit:
            v, unit = left*rval, "%s"%(runit)
        else:
            v, unit = left*rval, ''

    return right.__qbuilder__(v, unit)     
_shared_funcs["__rmul__"] = __rmul__
del __rmul__

# def __rmul__(right, left):        
#     if right.unit:
#         v, unit = left*right._value, right.unit
#     else:
#         v, unit = left*right._value, ''

#     return right.__qbuilder__(v,unit)  

def __pow__(self, exp):        
    if self.unit:
        sunit = parentize(self.unit)
        v, unit = self._value**exp , "%s**%s"%(sunit, exp)        
    else:
        v, unit = self._value**exp , ''    

    return self.__qbuilder__(v,unit)
_shared_funcs["__pow__"] = __pow__
del __pow__

def __add__(self, right):
    lv, rv, unit = _linear_op_prepare(self, right)
    return self.__qbuilder__(lv+rv, unit)
_shared_funcs["__add__"] = __add__
del __add__


def __radd__(self, left):
    lv, rv, unit = _linear_op_prepare(left, self)
    return self.__qbuilder__(lv+rv, unit)
_shared_funcs["__radd__"] = __radd__
del __radd__


def __le__(self, right):    
    lv, rv = _compare_op_prepare(self, right)
    return lv<=rv    
_shared_funcs["__le__"] = __le__
del __le__


def __lt__(self, right):    
    lv, rv = _compare_op_prepare(self, right)
    return lv<rv    
_shared_funcs["__lt__"] = __lt__
del __lt__

def __ge__(self, right):    
    lv, rv = _compare_op_prepare(self, right)
    return lv>=rv    
_shared_funcs["__ge__"] = __ge__
del __ge__

def __gt__(self, right):    
    lv, rv = _compare_op_prepare(self, right)
    return lv>rv    
_shared_funcs["__gt__"] = __gt__
del __gt__

def __eq__(self, right):    
    lv, rv = _compare_op_prepare(self, right)
    return lv==rv    
_shared_funcs["__eq__"] = __eq__
del __eq__



def __neg__(self):
    return self.__qbuilder__( -self._value, self.unit)
_shared_funcs["__neg__"] = __neg__
del __neg__

def __pos__(self):
    return self
_shared_funcs["__pos__"] = __pos__
del __pos__    

# def __mod__(self, m):
#     lv, rv, unit = _linear_op_prepare(self, m)
#     return self.__qbuilder__(lv%rv, unit)    
# _shared_funcs["__mod__"] = __mod__
# del __mod__


# def __rmod__(self, m):
#     lv, rv, unit = _linear_op_prepare(m, self)
#     return self.__qbuilder__(lv%rv, unit)           
# _shared_funcs["__rmod__"] = __rmod__
# del __rmod__
    
def __sub__(self, right):
    lv, rv, unit = _linear_op_prepare(self, right)
    return self.__qbuilder__(lv-rv, unit)
_shared_funcs["__sub__"] = __sub__
del __sub__


def __rsub__(self, left):
    lv, rv, unit = _linear_op_prepare(left, self)
    return self.__qbuilder__(lv-rv, unit)
_shared_funcs["__rsub__"] = __rsub__
del __rsub__

# def __div__(left, right):
#     runit = getunit(right)
#     if runit is not None:
#     #if isinstance(right, tuple(left._ref_classes)):
#         runit = right.unit
#         runit = "(%s)"%runit if any(o in runit for o in "*/+-") else runit
#         if left.unit:                
#             v, unit = left._value/right._value, "%s/%s"%(left.unit, runit)
#         else:
#             v, unit = left._value/right._value, "%s/%s"%(left._value, runit)

#     else:
#         if left.unit:
#             v, unit = left._value/right, "%s"%(left.unit)
#         else:
#             v, unit = left._value/right, ''

#     return left.__qbuilder__(v,unit)    
# _shared_funcs["__div__"] = __div__
# del __div__

def __div__(left, right):
    runit = getunit(right)
    lunit = parentize(left.unit)
    lval = left._value

    if runit is not None:        
        runit = parentize(runit)

        try:
            rval = right._value
        except:
            # this happen for units without value
            return NotImplemented
        
        if lunit:
            v, unit = lval/rval, "%s/%s"%(lunit, runit)
        else:
            v, unit = lval/rval, "%s/%s"%(lval, runit)    
    else:
        if lunit:
            v, unit = lval/right, "%s"%(lunit)
        else:
            v, unit = lvalue/right, ''
    return left.__qbuilder__(v, unit)
_shared_funcs["__div__"] = __div__
del __div__

def __floordiv__(left, right):
    runit = getunit(right)
    lunit = parentize(left.unit)
    lval = left._value

    if runit is not None:        
        runit = parentize(runit)

        try:
            rval = right._value
        except:
            # this happen for units without value
            return NotImplemented
        
        if lunit:
            v, unit = lval//rval, "%s/%s"%(lunit, runit)
        else:
            v, unit = lval//rval, "%s/%s"%(lval, runit)    
    else:
        if lunit:
            v, unit = lval//right, "%s"%(lunit)
        else:
            v, unit = lvalue//right, ''
    return left.__qbuilder__(v, unit)
_shared_funcs["__floordiv__"] = __floordiv__
del __floordiv__



def __rdiv__(right, left):       
    #if isinstance(left, tuple(right._ref_classes)):
    lunit = getunit(left)
    runit = parentize(right.unit)
    rval  = right._value 
    if lunit is not None:
        lunit = parentize(lunit)
        try:
            lval = left._value
        except:
            # this happen for units without value
            return NotImplemented

        if runit:
            v, unit = lval/rval, "%s/%s"%(lunit, runit)
        else:
            v, unit = lval/rval, "%s/%s"%(lval, runit)    
    else:
        if runit:
            v, unit = left/rval, "%s"%(runit)
        else:
            v, unit = left/rval, ''

    return right.__qbuilder__(v, unit)     
_shared_funcs["__rdiv__"] = __rdiv__
del __rdiv__


def __rfloordiv__(right, left):       
    #if isinstance(left, tuple(right._ref_classes)):
    lunit = getunit(left)
    runit = parentize(right.unit)
    rval  = right._value 
    if lunit is not None:
        lunit = parentize(lunit)
        try:
            lval = left._value
        except:
            # this happen for units without value
            return NotImplemented

        if runit:
            v, unit = lval/rval, "%s/%s"%(lunit, runit)
        else:
            v, unit = lval/rval, "%s/%s"%(lval, runit)    
    else:
        if runit:
            v, unit = left/rval, "%s"%(runit)
        else:
            v, unit = left/rval, ''

    return right.__qbuilder__(v, unit)     
_shared_funcs["__rfloordiv__"] = __rfloordiv__
del __rfloordiv__







def __repr__(self):
    return "%r [%s]"%(self.__tovalue__(self), self.unit)    
_shared_funcs["__repr__"] = __repr__
del __repr__ 

def __format__(self, format_spec):
    U = format_spec[-1]
    if U in ["U", "P"]:            
        format_spec = format_spec[:-1]
        if U == "U":
            unit = printofunit(self.R, self._unit)
        else:
            unit = self._unit

        if unit:
            return "{:{format_spec}} [{}]".format(self._value, unit, format_spec=format_spec)
        else:
            return "{:{format_spec}}".format(self._value, format_spec=format_spec)    
    else:
        return "{:{format_spec}}".format(self._value, format_spec=format_spec)

_shared_funcs["__format__"] = __format__
del __format__

def __qbuilder__(self, value, unit):
    return quantity(self.R, value, unit, QT=self.QT())
#_shared_funcs["__qbuilder__"] = staticmethod(quantity)
_shared_funcs["__qbuilder__"] = __qbuilder__
_shared_funcs["_unit"] = ""

####
# make a base class anyway 
_quantity_shared = type("_quantity_shared", tuple(), _shared_funcs)

def _prepare_quantity_class(cl):
    """ necessary to add the base func directly to the class and not adding 
    a subclass because it will cause a Segmentation fault: 11 when doing 
    e.g.  np.float32(quantity(np.float32(4.5)), "m")
    """
    if not hasattr(cl, "__tovalue__"):
        raise ValueError("a quantity class must define a '__tovalue__' function")

    for name, attr in _shared_funcs.iteritems():
        setattr(cl, name, attr)


class Qfloat(_quantity_shared, float):     
    def __new__(cl, value, unit):               
        value = float.__new__(cl, value)        
        value._unit = parseunit(cl.R, unit)
        value.__init_unit__()
        return value         

    @staticmethod
    def __tovalue__(v):
        return float(v)           

class Qint(_quantity_shared, int):                
    def __new__(cl, value, unit):    
        value = int.__new__(cl, value)        
        value._unit = parseunit(cl.R, unit)
        value.__init_unit__()
        return value       

    @staticmethod
    def __tovalue__(v):
        return int(v)           

def _parse_int(v):
    if isinstance(v,(int,long,bool)):
        return 
    raise TypeError()

class Qany(_quantity_shared, object):    
    def __new__(cl, value, unit):
        return quantity(value, unit)

class Unit(_quantity_shared, object):
    """ A unit that does not hold value """    
    def __new__(cl, unit):
        new = object.__new__(cl)
        new._unit = unit
        new.__init_unit__()
        return new

    def __repr__(self):
        return "Unit('%s')"%self.unit
    
    def new(self, value):
        return quantity(self.R, value, self.unit, QT=self.QT())    

    @classmethod
    def unitary_quantity(cl,unit):
        return cl(unit)    
        
    @staticmethod    
    def __tovalue__(v):
        raise NotImplementedError("Not defined value type")    
    
    @property
    def _value(self):
        raise NotImplementedError("This unit has no defined value")
         
    def __mul__(self, right):        
        if isinstance(right, self.__class__):            
            return self.__class__(parsetrueunit(self.R, "%s*%s"%(self.unit,right.unit)))

        return clone(right, self.unit, QT=self.QT())*right        
    
    #def __rmul__(self, left):
        return  NotImplemented

    def __rmul__(self, left):        
        if isinstance(left, self.__class__):
            return self.__class__(parsetrueunit(self.R, "%s*%s"%(left.unit, self.unit)))        
        return left*clone(left, self.unit, QT=self.QT())          
    
    def __pow__(self, exp):
        if isinstance(exp, self.__class__):                
            raise ValueError("x**y not allowed when x and y are quantity without value")
        if hasattr(exp, "__iter__"):
            raise ValueError("unit**y, y must be a scalar")
        return self.__class__(parsetrueunit(self.R, "%s**%s"%(parentize(self.unit), exp)))

    def __div__(self, right):        
        if isinstance(right, self.__class__):            
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(self.unit),parentize(right.unit))))

        return clone(right, self.unit, QT=self.QT())/right        
    
    #def __rmul__(self, left):
        return  NotImplemented

    def __rdiv__(self, left):        
        if isinstance(left, self.__class__):
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(left.unit), parentize(self.unit))))        
        return left/clone(left, self.unit, QT=self.QT())       

    def __floordiv__(self, right):        
        if isinstance(right, self.__class__):            
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(self.unit),parentize(right.unit))))

        return clone(right, self.unit, QT=self.QT())//right        
    
    #def __rmul__(self, left):
        return  NotImplemented

    def __rfloordiv__(self, left):        
        if isinstance(left, self.__class__):
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(left.unit), parentize(self.unit))))        
        return left//clone(left, self.unit, QT=self.QT())           

    def __neg__(self):
        raise ValueError("-unit not allowed when unit is a quantity without value")
            
    def __rpow__(self, exp):        
        raise ValueError("x**unit not allowed ")

    # def __add__(self, o):
    #     raise ValueError("Cannot add or substract quantity unit without value")     

    # def __sub__(self, o):
    #     raise ValueError("Cannot add os substract unit without value")     
    
    # def __rsub__(self, o):
    #     raise ValueError("Cannot add os substract unit without value")     
      
##  
# the valid types for a quantity are recorded in this lookup  
# the key is a parser __callable__ and item is the appropriate quantity class
# if the parser raise a TypeError, try with the next one.



QuantityTypes.__register_type__(float,Qfloat)
QuantityTypes.__register_type__(int,Qint)
QuantityTypes.__register_type__(type(None),Unit)












