from __future__ import print_function
from .api import (kindofunit, unitofunit, scaleofunit, 
                  basescaleofunit, dimensionsofunit,
                  printofunit,                                    
                  unitof, kindof, isunitless, valueof, 
                  scaleof, baseofunit, arekindconvertible, 
                  nameofunit, convert, 
                  UnitError, unitlist,
                  Registery, unitcode, kindcode
                )

from .registery import QuantityTypes
from string import Formatter

__all__ = ["Qfloat", "Qint", "quantity", "Qany", "Unit"]
try:
    unicode #python 2
except NameError:
    unicode = str #python 3
    basestring = (str,bytes)

Rglobal = Registery()
Rglobal.__isglobal__ = True
QTglobal = QuantityTypes(Rglobal)



#########################################################

def vectorize(func, val):
    if isinstance(val, basestring):
        return func(val)
    if hasattr(val, "__iter__"):
        return list(map(func, val))
    return func(val)

def vectorizeR(func, R, val):
    if isinstance(val, basestring):
        return func(R, val)
    if hasattr(val, "__iter__"):
        return list(map(lambda X:func(R,X), val))
    return func(R, val)

def _getunit(o):
    if isinstance(o,basestring):
        return o
    return getattr(o, "unit", None)

def getunit(o):
    """ return unit of object or None"""
    #if not hasattr(o, "value"):
    #   return None    
    return vectorize(_getunit, o)

def getvalue(o):
    return getattr(o, "value", o)

def getkind(o):
    """ return kind of object or None"""
    if not hasattr(o, "unit"):
        return None   
    return kindofunit(o.unit)

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
            return left.to(unitof(right)).value, right.value, unit

    elif mode == "keep left":
        def same_kind(left, right):
            same_kind_msg(left, right)
            unit = unitof(left)
            if error is "warning":
                print("  unit will be '%s'"%unit)
            return left.value, right.to(unitof(left)).value, unit    

    elif mode == "keep lowest":
        def same_kind(left, right):
            same_kind_msg(left, right)
            lscale, rscale = scaleofunit(left.R, unitof(left)), scaleofunit(right.R, unitof(right))
            if lscale>rscale:
                unit = unitof(right)
                if error is "warning":
                    print("  unit will be '%s'"%unitof(right))
                return left.to(unitof(right)).value, right.value, unit
            else:
                unit = unitof(left)                
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left.value, right.to(unitof(left)).value, unit

    elif mode == "keep highest":
        def same_kind(left, right):
            same_kind_msg(left, right)
            lscale, rscale = scaleofunit(left.R, unitof(left)), scaleofunit(right.R, unitof(right))
            if lscale<rscale:
                unit = unitof(right)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left.to(unitof(right)).value, right.value, unit
            else:
                unit = unitof(left)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left.value, right.to(unitof(left)).value, unit

    elif mode == "keep base":
        def same_kind(left, right):
            same_kind_msg(left, right)
            lscale, rscale = scaleofunit(left.R, unitof(left)), scaleofunit(right.R, unitof(right))
            if abs(lscale-1)>abs(rscale-1):
                unit = unitof(right)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left.to(unitof(right)).value, right.value, unit
            else:
                unit = unitof(left)
                if error is "warning":
                    print("  unit will be '%s'"%unit)
                return left.value, right.to(unitof(left)).value, unit    


    elif mode == "drop":            
        def same_kind(left, right):            
            same_kind_msg(left, right)
            if error is "warning":
                    print("  unit is dropped")
            return left.value, right.value, ''
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
    else:
        def unitless_kind_msg(left, right):
            return 
            
    if mode is "keep":
        def unitless_kind(left, right):
                unitless_kind_msg(left, right)            
                return valueof(left), valueof(right), left.unit if hasattr(left, "unit") else right.unit
    else:
        def unitless_kind(left, right):
            unitless_kind_msg(left, right)
            return valueof(left), valueof(right), ''

unitless_operation_setting("silent", "drop")            

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
          
    #if isinstance(unit, basestring):
    #    unit = parsetrueunit(R, unit)
    #else:
    #    unit = [parsetrueunit(R, u) for u in unit]
     
    if issubclass(quantityclass,BaseUnit):
        return quantityclass(unit)          

    return quantityclass(value, unit)

def clone(value, unit, QT=QTglobal):
    """ clone a value to a unitary quantity object with the given unit 
    and the class of value
    """
    cl, v = QT.get_quantity_class(value)

    if cl is None:
        raise ValueError("Cannot make a quantity with value of type '%s'"%type(value))

    if hasattr(cl, "unitary_quantity"):
        return v, cl.unitary_quantity(unit)
    else:        
        return v, cl(1.0, unit)    

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

##################################################################
#
#   A quantity Formater 
#
##################################################################
class QuantityFormatter(Formatter):
    def convert_field(self, value, conversion):
        if conversion=="U":
            return value.unit
        else:
            return Formatter.convert_field(self, value, conversion)    
quantityformatter = QuantityFormatter().format

######################################################################
#
# We define here the base quantity functions separatly because it appears
# that subclassing in numpy.floatxx or numpy.intxx makes segmentation fault 11
# 
######################################################################

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

class ConvertorInstance(object):
    def __init__(self, convertor, cl, obj):        
        self.convertor = convertor
        self.cl  = cl
        self.obj = obj        
    
    @property
    def _register(self):
        if self.obj is None:
            return self.cl.QT().R
        else:
            return self.obj.QT().R    

    def __getitem__(self, item):
        if isinstance(item, tuple):
            return list(self(i) for i in item)
        return self(item)
                        
    def __call__(self, *args, **kwargs):
        if len(args)>2:
            raise ValueError("to takes zero to two argument")
        R = self._register

        if len(args)==3:
            system = args[-1]
            args = args[:-1]
            if "system" in kwargs:
                raise TypeError("got multiple values for keyword argument 'system'")
        system = kwargs.get("system", None)        
                
        if not args:
            unit = None
        elif len(args)==2:
            v,u = args
            if issubclass(self.cl, BaseUnit):                
                return vectorize(lambda X: quantity(R, v, X, self.cl.QT()), u)
                #return quantity(R, v, u, self.cl.QT())
            else:    
                return vectorize(lambda X: self.cl(v,X), u)
                #return self.cl(*args)        
        else:
            unit, = args        

        
        if self.obj is None:
            if unit is None:
                raise ValueError("provide a unit")
            elif not isinstance(unit, basestring):
                
                unit = getunit(unit)

                if unit is None:
                    raise ValueError("new unit must be a string or must have the 'unit' atribute")

            #unit = vectorize(lambda X:unitofunit(R,X)[0], unit or '')
            unit = vectorize(lambda X:X, unit or '')
            if hasattr(self.cl, "unitary_quantity"):
                return vectorize(lambda X:self.cl.unitary_quantity(X), unit)
                #return self.cl.unitary_quantity(unit)
            else:    
                return vectorize(lambda X:self.cl(1.0,X), unit)#self.cl(1.0, unit)

        if unit is None and system is None:
            unit = vectorize(lambda X:unitofunit(self.obj.R,X)[0], self.obj.unit)
                    
        #unit = vectorize(lambda X:unitofunit(self.obj.R,X)[0], unit)
        
        if isinstance(self.obj, BaseUnit):
            return vectorize(lambda X,system=system:tofunc(self.obj.new(1.0),X,system=system), unit)
            #return  tofunc(self.obj.new(1.0), unit)

        return tofunc(self.obj, unit, system=system)

    def __getattr__(self, attr):
        return self.__call__(attr)
    
    def new(self, value, unit=None):
        if unit is None:
            if self.obj is None:
                return self.cl(value)
            unit, _ = unitofunit(self._register, self.obj.unit) 
            return self.cl(value, unit)
                    
        unit, _ = unitofunit(self._register, unit)            
        return self.cl(value, unit)

    @property
    def base(self):
        if self.obj is None:
            raise Exception("'to' called out of object context")
        return self.__call__(baseofunit(self._register, self.obj.unit))
            
    def import_units(self, setter, *args):       
        units = []
        if not len(args):
            units = list(self.iterunits())
        else:                            
            for a in args:
                units.extend(s.strip() for s in a.split(","))
            
        for u in units:
            setter(u, self(u))
        return units
    
    def import_units_of_kinds(self, setter, *args):
        if not len(args):
            return import_units(setter)
        kinds = []    
        for a in args:
            kinds.extend(s.strip() for s in a.split(","))                     

        units = []    
        for kind in kinds:
            for u in self.iterunits(kind):
                units.append(u)
                setter(u, self(u))
        return units

    def iterunits(self, kind=None):
        return self._register.iterunits(kind)

    def iterkinds(self):
        return self._register.iterkinds()        


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

def tofunc(quantity, newunit, system=None):
    """ convert the quantity to a new unit e.g 'm' to 'km'"""
    if not quantity.unit :
        raise ValueError('quantity has no unit')
    
    if not isinstance(newunit, basestring) and system is None:
        #newunit = getunit(newunit)
        #if newunit is None:
        if hasattr(newunit, "unit"):
            newunit = newunit.unit
        elif hasattr(newunit, "__iter__"):
            newunit = getunit(newunit)            
        else:            
            raise ValueError("new unit must be a string, a iterable or must have the 'unit' atribute got %s"%newunit)

    ## save sometime return quantity if the unit is unchanged
    ## should not be a problem because quantity is un-mutable
    #if newunit == quantity.unit:
    #    return quantity

    newcl =  quantity.__class__ if not hasattr(quantity, "__qwrapper__") else quantity.__qwrapper__    

    if hasattr(quantity, "__qconvertor__"):
        return quantity.__qconvertor__(quantity.R, quantity.value, quantity.unit, newunit, system=system, inside=newcl)    
        
    if hasattr(quantity.unit,"__iter__"):
        return [convert(quantity.R, quantity.value, u, 
                        newunit, inside=newcl) for u in quantity.unit]

    return convert(quantity.R, quantity.__tovalue__(quantity), quantity.unit, newunit, system=system, inside=newcl)



class _QuantityShared_:
    to = Convertor()
    R  = Rglobal


    def inplace_convert(self,  unit):
        raise ValueError("convert works only on mutable object")

    @staticmethod
    def _QTwr_():
        return None

    @classmethod    
    def QT(cl):
        return cl._QTwr_() or QTglobal        

    def new(self, value):
        """ build a new quantity of same unit with new value """
        return self.__class__(value, self.unit) 

    @property    
    def unit(self):
        """ quantity unit """
        ## use __getattribute__ to avoid problems with 
        ## too fancy objects 
        #return object.__getattribute__(self, "_unit")
        return self._unit

    @property
    def value(self):
        """ float representation of quantity """
        #return object.__getattribute__(self, "__tovalue__")(self)
        return self.__tovalue__(self)

    @property
    def unitname(self):
        return vectorizeR(nameofunit, self.R, self.unit)
        return nameofunit(self.R, self.unit)
    @property
    def unitkind(self):
        """ quantity unit kind value """
        return vectorizeR(kindofunit, self.R, self.unit)
        return kindofunit(self.R, self.unit)
    @property    
    def unitscale(self):
        """ quantity scale compare to the base unit of its kind """
        return vectorizeR(basescaleofunit, self.R, self.unit)        
        return basescaleofunit(self.R, self.unit)    
    @property      
    def unitdimensions(self):
        """ dimensions of unit """
        return vectorizeR(dimensionsofunit, self.R, self.unit)        
        return dimensionsofunit(self.R, self.unit)    

    def tobase(self):
        """ quantity transform to its kind base unit """
        return vectorize(self.to, vectorizeR(baseofunit,self.R, self.unit))    
        return self.to(self.unitbase)

    @classmethod    
    def unitary_quantity(cl, unit):
        """ quantity scale value """
        return vectorize(lambda X:cl(1.0,X), unit)
        return cl(1.0, unit)
    @property      
    def unitbase(self):
        """ The string unit coresponding to the base of its kind """
        return vectorizeR(baseofunit,self.R, self.unit)
        return baseofunit(self.R, self.unit)

    def __init_unit__(self, unit):        
        if isinstance(unit, basestring):
            self._unit = unitcode(self.R, unit)
        elif hasattr(unit, "unit"):
            self._unit = unitcode(self.R, unit.unit)
        else:
            raise ValueError("Expecting a string or an object with 'unit' attribute,  got %r"%unit)            

    def __mul__(left, right):
        runit = getattr(right, "unit", None)
        lunit = left.unit
        lval = left.value
        #if isinstance(right, tuple(left._ref_classes)):
        if runit:
            try:
                rval = right.value
            except:
                # this happen for units without value
                return NotImplemented
                                            
            if left.unit:
                v, unit = lval*rval, "%s*%s"%(lunit, right.unit)
            else:
                v, unit = lval*rval, "%s*%s"%(lval, right.unit)    
        else:
            rval = left.QT().parsevalue(right)
            if lunit:
                v, unit = lval*rval, "%s"%(lunit)
            else:
                v, unit = lval*rval, ''
        return left.__qbuilder__(v, unit)

    def __rmul__(right, left):       
        #if isinstance(left, tuple(right._ref_classes)):        
        lunit = getattr(left, "unit", None)
        runit = right.unit
        rval  = right.value 
        if lunit:
            try:
                lval = left.value
            except:
                # this happen for units without value
                return NotImplemented
            

            if runit:
                v, unit = lval*rval, "%s*%s"%(lunit, runit)
            else:
                v, unit = lval*rval, "%s*%s"%(lval, runit)    
        else:
            lval = right.QT().parsevalue(left)
            if runit:
                v, unit = lval*rval, "%s"%(runit)
            else:
                v, unit = lval*rval, ''

        return right.__qbuilder__(v, unit)     

    def __pow__(self, exp):        
        if self.unit:
            sunit = parentize(self.unit)
            v, unit = self.value**exp , "%s**%s"%(sunit, exp)        
        else:
            v, unit = self.value**exp , ''    

        return self.__qbuilder__(v,unit)


    def __add__(self, right):
        lv, rv, unit = _linear_op_prepare(self, right)
        return self.__qbuilder__(lv+rv, unit)


    def __radd__(self, left):
        lv, rv, unit = _linear_op_prepare(left, self)
        return self.__qbuilder__(lv+rv, unit)

    def __le__(self, right):    
        lv, rv = _compare_op_prepare(self, right)
        return lv<=rv    


    def __lt__(self, right):    
        lv, rv = _compare_op_prepare(self, right)
        return lv<rv    

    def __ge__(self, right):    
        lv, rv = _compare_op_prepare(self, right)
        return lv>=rv    

    def __gt__(self, right):    
        lv, rv = _compare_op_prepare(self, right)
        return lv>rv    

    def __eq__(self, right):    
        lv, rv = _compare_op_prepare(self, right)
        return lv==rv    

    def __neg__(self):
        return self.__qbuilder__( -self.value, self.unit)

    def __pos__(self):
        return self

    def __imul__(self, scl):
        return self * scl

    def __idiv__(self, scl):
        return self / scl
    
    def __ifloordiv__(self, scl):
        return self // scl        
    

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

    def __rsub__(self, left):
        lv, rv, unit = _linear_op_prepare(left, self)
        return self.__qbuilder__(lv-rv, unit)

    def __div__(left, right):
        runit = getattr(right, "unit", None)
        lunit = parentize(left.unit)
        lval = left.value

        if runit:        
            runit = parentize(runit)

            try:
                rval = right.value
            except:
                # this happen for units without value
                return NotImplemented
            
            if lunit:
                v, unit = lval/rval, "%s/%s"%(lunit, runit)
            else:
                v, unit = lval/rval, "1/%s"%(runit)    
        else:
            rval = left.QT().parsevalue(right)
            if lunit:
                v, unit = lval/rval, "%s"%(lunit)
            else:
                v, unit = lvalue/rval, ''           
        return left.__qbuilder__(v, unit)

    __truediv__ = __div__

    def __floordiv__(left, right):
        runit = getattr(right, "unit", None)
        lunit = parentize(left.unit)
        lval = left.value

        if runit:        
            runit = parentize(runit)

            try:
                rval = right.value
            except:
                # this happen for units without value
                return NotImplemented
            
            if lunit:
                v, unit = lval//rval, "%s/%s"%(lunit, runit)
            else:
                v, unit = lval//rval, "1/%s"%(runit)    
        else:
            rval = left.QT().parsevalue(right)
            if lunit:
                v, unit = lval//rval, "%s"%(lunit)
            else:
                v, unit = lvalue//rval, ''
        return left.__qbuilder__(v, unit)


    def __rdiv__(right, left):       
        #if isinstance(left, tuple(right._ref_classes)):
        lunit = getattr(left, "unit", None)
        runit = parentize(right.unit)
        rval  = right.value 
        if lunit:
            lunit = parentize(lunit)
            try:
                lval = left.value
            except:
                # this happen for units without value
                return NotImplemented

            if runit:
                v, unit = lval/rval, "%s/%s"%(lunit, runit)
            else:
                v, unit = lval/rval, "%s"%(lunit)    
        else:
            lval = right.QT().parsevalue(left)
            if runit:
                v, unit = lval/rval, "1/%s"%(runit)
            else:
                v, unit = lval/rval, ''

        return right.__qbuilder__(v, unit)     

    def __rfloordiv__(right, left):       
        #if isinstance(left, tuple(right._ref_classes)):
        lunit = getattr(left, "unit", None)
        runit = parentize(right.unit)
        rval  = right.value 
        if lunit:
            lunit = parentize(lunit)
            try:
                lval = left.value
            except:
                # this happen for units without value
                return NotImplemented

            if runit:
                v, unit = lval/rval, "%s/%s"%(lunit, runit)
            else:
                v, unit = lval/rval, "%s"%(lunit)    
        else:
            lval = right.QT().parsevalue(left)
            if runit:
                v, unit = lval/rval, "1/%s"%(runit)
            else:
                v, unit = lval/rval, ''

        return right.__qbuilder__(v, unit)     

    def __repr__(self):
        return "%r %s"%(self.value, self.unit)    

    def __str__(self):        
            return "%r %s"%(self.value, printofunit(self.R,self.unit))    
    

    def __format__(self, format_spec):
        if not format_spec:
            return str(self.value)

        U = format_spec[-1]
        if U in ["U", "R", "N"]:            
            format_spec = format_spec[:-1]
            

            if U == "U":
                unit = printofunit(self.R, self.unit)
            elif U == "N":
                unit = nameofunit(self.R, self.unit)
            else:
                unit = self.unit

            if format_spec:    
                if unit:
                    return "{:{format_spec}} {}".format(self.value, unit, format_spec=format_spec)
                else:
                    return "{:{format_spec}}".format(self.value, format_spec=format_spec)
            else:
                if unit:
                    return "{}".format(unit, format_spec=format_spec)
                else:
                    return ""
        else:
            return "{:{format_spec}}".format(self.value, format_spec=format_spec)

    @classmethod
    def __qconvertor__(cl,R, value, unit, newunit=None, system=None, inside=None):
        return convert(R, value, unit, newunit, system=system, inside=inside)

        
    def __qbuilder__(self, value, unit):
        return quantity(self.R, value, unit, QT=self.QT())

def _prepare_quantity_class(cl):
    """ necessary to add the base func directly to the class and not adding 
    a subclass because it will cause a Segmentation fault: 11 when doing 
    e.g.  np.float32(quantity(np.float32(4.5)), "m")
    """
    if not hasattr(cl, "__tovalue__"):
        raise ValueError("a quantity class must define a '__tovalue__' function")

    _shared_funcs = dict(_QuantityShared_.__dict__)
    for k in ["__module__", "__dict__", "__doc__", "__name__"]:
        _shared_funcs.pop(k,None)

    for name, attr in _shared_funcs.items():
        setattr(cl, name, attr)

class Qfloat(_QuantityShared_, float):     
    def __new__(cl, value, unit):               
        value = float.__new__(cl, value)
        value.__init_unit__(unit)
        return value         

    @staticmethod
    def __tovalue__(v):
        return float(v)           

class Qint(_QuantityShared_, int):                
    def __new__(cl, value, unit):    
        value = int.__new__(cl, value)
        value.__init_unit__(unit)
        return value       

    @staticmethod
    def __tovalue__(v):
        return int(v)           

def _parse_int(v):
    if isinstance(v,(int,long,bool)):
        return 
    raise TypeError()

class Qany(_QuantityShared_, object):    
    def __new__(cl, value, unit):
        return quantity(value, unit)

class BaseUnit(_QuantityShared_):
    """ A unit that does not hold value """    
    
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
    def value(self):
        raise ValueError("This unit has no defined value")
         
    def __mul__(self, right):        
        if isinstance(right, self.__class__):            
            return self.__class__(parsetrueunit(self.R, "%s*%s"%(self.unit,right.unit)))

        
        right, left = clone(right, self.unit, QT=self.QT())            
        return left*right        
        

    def __rmul__(self, left):        
        if isinstance(left, self.__class__):
            return self.__class__(parsetrueunit(self.R, "%s*%s"%(left.unit, self.unit)))
        
        left, right = clone(left, self.unit, QT=self.QT())                       
        return left*right           
    
    def __pow__(self, exp):
        if isinstance(exp, self.__class__):                
            raise ValueError("x**y not allowed when x and y are quantity without value")
        if hasattr(exp, "__iter__"):
            raise ValueError("unit**y, y must be a scalar")
        return self.__class__(parsetrueunit(self.R, "%s**%s"%(parentize(self.unit), exp)))

    def __div__(self, right):        
        if isinstance(right, self.__class__):            
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(self.unit),parentize(right.unit))))
        right, left = clone(right, self.unit, QT=self.QT())    
        return left/right    

    __truediv__ = __div__    
        
    #def __rmul__(self, left):
    #    return  NotImplemented

    def __rdiv__(self, left):        
        if isinstance(left, self.__class__):
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(left.unit), parentize(self.unit))))        
        left, right = clone(left, self.unit, QT=self.QT())            
        return left/right   

    def __floordiv__(self, right):        
        if isinstance(right, self.__class__):            
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(self.unit),parentize(right.unit))))
        right, left = clone(right, self.unit, QT=self.QT())
        return left//right        
    
    #def __rmul__(self, left):
        return  NotImplemented

    def __rfloordiv__(self, left):        
        if isinstance(left, self.__class__):
            return self.__class__(parsetrueunit(self.R, "%s/%s"%(parentize(left.unit), parentize(self.unit))))        
        left, right = clone(left, self.unit, QT=self.QT())                
        return left//right           

    def __neg__(self):
        raise ValueError("-unit not allowed when unit is a quantity without value")
            
    def __rpow__(self, exp):        
        raise ValueError("x**unit not allowed ")

    def __imul__(self, scl):
        return self * scl

    def __idiv__(self, scl):
        return self / scl
    
    def __ifloordiv__(self, scl):
        return self // scl        
                

class Unit(BaseUnit, object):
    def __new__(cl, unit):
        new = object.__new__(cl)
        new.__init_unit__(unit)
        return new   


QuantityTypes.__register_type__(float,Qfloat)
QuantityTypes.__register_type__(int,Qint)
QuantityTypes.__register_type__(type(None),Unit)












