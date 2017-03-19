from . import base
from .base import (_QuantityShared_,                     
                    _prepare_quantity_class, 
                    parseunit, parsetrueunit,
                    QuantityTypes, parentize,
                    convert, unitcode, unitlist, 
                    _linear_op_prepare
                  )
from . import api
from .base import BaseUnit
import numpy as np
import sys

__all__ = ["Qarray", "Qfloat128", "Qfloat64", "Qfloat32", "Qfloat16", 
           "Qint64", "Qint32", "Qint16", "Qint8", 
           "Quint64", "Quint32", "Quint16", "Quint8",
           "Qcomplex256", "Qcomplex128", "Qcomplex64", 
           "Unit"
        ]

def isarray(v):
    """ to check if an array of scalar """
    return hasattr(v, "__iter__")

def parserecord(a):    
    a.dtype.type == np.record
    return a

def im_func(o):
    try:
        return o.im_func
    except AttributeError:
        return o.__func__

class Unit(BaseUnit, np.ndarray):
    def __new__(cl, unit):
        new = np.asarray(1.0).view(cl)        
        new.__init_unit__(unit)
        return new

    def __array_wrap__(self, o):
        if isinstance(o, np.ndarray) and not o.shape:
            return self.__qbuilder__(o.dtype.type(o), self.unit)
        return self.__qbuilder__(o, self.unit)

####
# So far a None type is registered to base.Unit 
# we need to remove it a nd replace it by the numpy version
QuantityTypes.__register_type__(type(None), Unit, replace=type(None))


class _QaInit_:
    def __init_unit__(self, unit):
        if isinstance(unit, basestring):
            if "," in unit:
                self._unit = unitlist(unit.split(","))
            else:                
                self._unit = unitcode(self.R, unit)

        elif hasattr(unit, "unit"):
            self._unit = unitcode(self.R, unit.unit)
        else:            
            if not hasattr(unit, "__iter__"):
                raise ValueError("Expecting a string, an object with 'unit' attribute or an iterable ,  got %r"%unit)                    
            self._unit = unitlist(self.R, (u if isinstance(u, basestring) else u.unit for u in unit))

class Qarray(_QuantityShared_, _QaInit_, np.ndarray):
    _unit = u''
    def __new__(subtype, data_array, unit=""):
        self = np.asanyarray(data_array)
        oview = type(self)        
        self = self.view(subtype)

        if isinstance(unit, basestring):
            self._sameunit = True
            unit = parseunit(self.R, unit)
        else:
            self._sameunit = False
            if len(unit) != len(self.dtype):
                raise ValueError("Number of array field is %d number of units is %s"%( len(self.dtype),len(unit)))
            unit = [parseunit(self.R, u) for u in unit]
                                
        #if self.dtype.type is not Qvoid and self.dtype.fields:
        #    self.dtype = np.dtype((Qvoid, self.dtype))
        #    self.dtype.type.unit = self.unit      

        self.__init_unit__(unit)    
        return self

    def __array_finalize__(self, obj):
        if issubclass(self.dtype.type, np.void):            
            if self.dtype.type is not Qvoid and self.dtype.fields:
                self.dtype = np.dtype((Qvoid, self.dtype))       


    
    def inplace_convert(self, unit, __scale__= None):
        if not self.shape:
            raise ValueError("convert works only on mutable object")        
        if self.dtype.fields:
            if not isinstance(unit, basestring):
                if hasattr(unit, "unit"):
                    unit = unit.unit
                    if hasattr(unit, "__iter__"):
                        # one more time 
                        unit = list(unit)
                    elif hasattr(self.unit, "__iter__"):
                        unit = [unit]*len(self.dtype.fields)

                elif hasattr(unit, "__iter__"):
                    unit = list(unit)
                else:
                    raise ValueError("unit expecting to be a string or object with 'unit' attribute, got %s"%unit)        
            else:
                # this is a string
                if "," in unit:
                    unit = unitlist(unit)
                elif hasattr(self.unit, "__iter__"):    
                    unit = [unit]*len(self.dtype.fields)        
            sunit = self.unit
            if isinstance(sunit, basestring):
                sunit = [sunit]*len(self.dtype.fields)         


            for u,nu,field in zip(sunit, unit,self.dtype.fields):
                scale = api.convert(self.R, 1.0, u, nu)
                if __scale__ is not None:
                    scale = scale * __scale__
                self[field] *= (scale)                
            self.__init_unit__(unit)  

            return None
        ###########################################


        if not isinstance(unit, basestring):
            if not hasattr(unit, "unit"):
                raise ValueError("unit expecting to be a string or object with 'unit' attribute, got %s"%unit)
            unit = unit.unit
        
        scale = api.convert(self.R, 1.0, self.unit, unit)
        if __scale__ is not None:
            scale = scale * __scale__
        np.ndarray.__imul__( self, scale)
        self.__init_unit__(unit)
        

    def __getitem__(self, item):
        obj = np.ndarray.__getitem__(self, item)

        # if isinstance(obj, np.ndarray):
        #     if obj.dtype.fields:
        #         obj = obj.view(type(self))
        #         if issubclass(obj.dtype.type, np.record):
        #             obj = obj.view(dtype=(self.dtype.type, obj.dtype))

        #         elif issubclass(obj.dtype.type, np.void):
        #             obj = obj.view(dtype=(self.dtype.type, obj.dtype))
        #     else:
        #         obj = obj.view(type=Qarray)
                

        if isinstance(item, basestring):
            unit = self.__unitoffield__(item)
        else:
            unit = self.unit         
        return self.__qbuilder__(obj, unit)    
                     
                    
        # try:
        #     len(sub)
        # except TypeError:
        #     return self.__qbuilder__(sub, unit)
        #     #return self.QT().get_quantity_class(type(sub))[0](sub, unit)
        #     #return self.__qbuilder__(sub, unit)            
        # else:            
        #     return self.__qbuilder__(np.asarray(sub), unit)
                
    def __getslice__(self, a,b): 
        sub = np.ndarray.__getslice__(self,a,b)       
        return Qarray(np.asarray(sub), self.unit)
    
    def __unitoffield__(self, item):
        if self._sameunit:
            return self.unit
        try:
            i = self.dtype.names.index(item)
        except ValueError:
            raise ValueError("no field of name %s"%item)
        return self.unit[i]           

    def __tovalue__(self, v):
        v = np.asarray(v, dtype=self.dtype)
        if issubclass(v.dtype.type, Qrecord):
            object.__setattr__(v, "dtype",np.dtype((np.record, v.dtype))) 
        if issubclass(v.dtype.type, Qvoid):
            object.__setattr__(v, "dtype",np.dtype((np.void, v.dtype))) 
        return v
    @property        
    def __qwrapper__(self):
        return lambda a,u: Qarray(np.asanyarray(a, self.dtype), u) 

    @classmethod
    def __qconvertor__(cl, R, value, unit, newunit, newcl):
        if isinstance(value, np.ndarray):

            if value.dtype.fields and hasattr(unit,"__iter__"):
                if not hasattr(newunit, "__iter__"):
                    newunit = [newunit] * len(value.dtype.fields)

                out = np.ones_like(value)
                for i,(field,nu) in enumerate(zip(value.dtype.names,newunit)):
                    out[field] = convert(R, value[field], unit[i], nu, newcl)
                return out    

        return convert(R, value, unit, newunit, newcl)
            
    def __array_wrap__(self, o):
        if isinstance(o, np.ndarray) and not o.shape:
            return self.__qbuilder__(o.dtype.type(o), self.unit)
        return self.__qbuilder__(o, self.unit)

    def __imul__(self, scl):
        #self = np.ndarray.__imul__(self, getattr(scl,"value", scl))
        sunit = self.unit
        self = np.ndarray.__imul__(self, scl)
        if hasattr(scl, "unit"):
            if sunit:
                self.__init_unit__(parsetrueunit(self.R, "%s*%s"%(sunit,scl.unit)))
            else:
                self.__init_unit__(parsetrueunit(self.R, (scl.unit)))
        elif sunit:
            self.__init_unit__(sunit)

        return self    
    
    def __idiv__(self, scl):
        #self = np.ndarray.__imul__(self, getattr(scl,"value", scl))
        sunit = self.unit        
        self = np.ndarray.__idiv__(self, scl)
        if hasattr(scl, "unit"):
            if sunit:
                self.__init_unit__(parsetrueunit(self.R, "%s/%s"%(parentize(sunit),parentize(scl.unit))))
            else:
                self.__init_unit__(parsetrueunit(self.R, (scl.unit)))
        elif sunit:
            self.__init_unit__(sunit)
                
        return self    
    
    def __ifloordiv__(self, scl):
        #self = np.ndarray.__imul__(self, getattr(scl,"value", scl))
        self = np.ndarray.__ifloordiv__(self, scl)
        if hasattr(scl, "unit"):
            if sunit:
                self.__init_unit__(parsetrueunit(self.R, "%s/%s"%(parentize(sunit),parentize(scl.unit))))
            else:
                self.__init_unit__(parsetrueunit(self.R, (scl.unit)))
        elif sunit:
            self.__init_unit__(sunit)                
        return self


    def __iadd__(self, offset):
        lv, rv, unit = _linear_op_prepare(self, offset)        
        self = np.ndarray.__iadd__(self, rv)
        self.__init_unit__(unit)
        return self
    
    def __isub__(self, offset):
        lv, rv, unit = _linear_op_prepare(self, offset)
        self = np.ndarray.__isub__(self, rv)
        self.__init_unit__(unit)
        return self

        
        





    
        

class Qfloat128(np.float128):
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.float128), unit)
        else:    
            new = np.float128.__new__(cl, v)        
            new.__init_unit__(unit)        
        return new

    @staticmethod
    def __tovalue__(v):
        return np.float128(v)

QuantityTypes.__register_type__(np.float128, Qfloat128)
_prepare_quantity_class(Qfloat128)

class Qfloat64(np.float64):
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.float64), unit)
        else: 
            new = np.float64.__new__(cl, v)            
            new.__init_unit__(unit)                
        return new

    @staticmethod
    def __tovalue__(v):
        return np.float64(v)

QuantityTypes.__register_type__(np.float64, Qfloat64)
_prepare_quantity_class(Qfloat64)



class Qfloat32(np.float32):    
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.float32), unit)
        else: 
            new = np.float32.__new__(cl, v)            
            new.__init_unit__(init)                 
        return new

    @staticmethod
    def __tovalue__(v):
        return np.float32(v)

    def __repr__(self):
        return "%r [%s]"%(self.__tovalue__(self), self.unit)    
                    
QuantityTypes.__register_type__(np.float32, Qfloat32)
_prepare_quantity_class(Qfloat32)


class Qfloat16(np.float16):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.float16), unit)
        else: 
            new = np.float16.__new__(cl, v)
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.float16(v)

QuantityTypes.__register_type__(np.float16, Qfloat16)
_prepare_quantity_class(Qfloat16)


class Qint64(np.int64):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.int64), unit)
        else: 
            new = np.int64.__new__(cl, v)
            new.__init_unit__(unit)
                 
        return new

    @staticmethod
    def __tovalue__(v):
        return np.int64(v)
QuantityTypes.__register_type__(np.int64, Qint64)
_prepare_quantity_class(Qint64)

class Qint32(np.int32):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.int32), unit)
        else: 
            new = np.int32.__new__(cl, v)
            new.__init_unit__(unit)         
        return new

    @staticmethod
    def __tovalue__(v):
        return np.int32(v)
QuantityTypes.__register_type__(np.int32, Qint32)
_prepare_quantity_class(Qint32)
#_prepare_quantity_class(Qint32)

class Qint16(np.int16):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.int16), unit)
        else: 
            new = np.int16.__new__(cl, v)
            new.__init_unit__(unit)

        return new

    @staticmethod
    def __tovalue__(v):
        return np.int16(v)
QuantityTypes.__register_type__(np.int16, Qint16)
_prepare_quantity_class(Qint16)

class Qint8(np.int8):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.int8), unit)
        else: 
            new = np.int8.__new__(cl, v)
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.int8(v)
QuantityTypes.__register_type__(np.int8, Qint8)
_prepare_quantity_class(Qint8)



class Quint64(np.uint64):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.uint64), unit)
        else: 
            new = np.uint64.__new__(cl, v)            
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.uint64(v)
QuantityTypes.__register_type__(np.uint64, Quint64)
_prepare_quantity_class(Quint64)

class Quint32(np.uint32):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.uint32), unit)
        else:
            new = np.uint32.__new__(cl, v)
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.uint32(v)
QuantityTypes.__register_type__(np.uint32, Quint32)
_prepare_quantity_class(Quint32)



class Quint16(np.uint16):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.uint16), unit)
        else:
            new = np.uint16.__new__(cl, v)            
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.uint16(v)
QuantityTypes.__register_type__(np.uint16, Quint16)
_prepare_quantity_class(Quint16)


class Quint8(np.uint8):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.uint8), unit)
        else:
            new = np.uint8.__new__(cl, v)            
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.uint8(v)
QuantityTypes.__register_type__(np.uint8, Quint8)
_prepare_quantity_class(Quint8)


class Qcomplex256(np.complex256):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.complex256), unit)
        else:
            new = np.complex256.__new__(cl, v)
            new.__init_unit__(unit)
        return new

    @staticmethod
    def __tovalue__(v):
        return np.complex256(v)
QuantityTypes.__register_type__(np.complex256, Qcomplex256)
_prepare_quantity_class(Qcomplex256)

class Qcomplex128(np.complex128):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.complex128), unit)
        else:
            new = np.complex128.__new__(cl, v)        
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.complex128(v)
QuantityTypes.__register_type__(np.complex128, Qcomplex128)
_prepare_quantity_class(Qcomplex128)


class Qcomplex64(np.complex64):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.complex64), unit)
        else:
            new = np.complex64.__new__(cl, v)
            new.__init_unit__(unit)            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.complex64(v)
QuantityTypes.__register_type__(np.complex64, Qcomplex64)
_prepare_quantity_class(Qcomplex64)

class Qvoid(_QuantityShared_, _QaInit_, np.void):
    def __new__(cl, data, unit):
        #new = data.view(Qvoid)        
        if not isinstance(data, np.void):
            new = Qarray(np.asarray(data), unit)
        else:             
            new = data.view(Qarray)                
            new.__init_unit__(unit)            
        return new 

    def __getitem__(self, item):
        obj = np.void.__getitem__(self, item)
        unit = self.__unitoffield__(item)
        return self.__qbuilder__(obj, unit)
    
    def __unitoffield__(self, item):
        if isinstance(self._unit, basestring):
            return self.unit
        try:
            i = self.dtype.names.index(item)
        except ValueError:
            raise ValueError("no field of name %s"%item)
        return self.unit[i]       

    @staticmethod
    def __tovalue__(v):        
        return v.view(np.recarray)

class Qrecord(_QuantityShared_, _QaInit_, np.record):
    def __new__(cl, data, unit):
        #new = data.view(Qrecord)
        if not isinstance(data, np.void):
            new = Qarray(np.asarray(data), unit)
        else:    
            new = data.view(Qrecarray)        
            # new = np.frombuffer( data.data, data.dtype)[0]
            #new.__class__ = cl
            new.__init_unit__(unit)                    
        return new

    @staticmethod
    def __tovalue__(v):
        return v.view(np.recarray)

    def __getitem__(self, item):
        obj = np.void.__getitem__(self, item)
        unit = self.__unitoffield__(item)
        return self.__qbuilder__(obj, unit)

    def __getattribute__(self, attr):        
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        return self[attr]
                            
    def __unitoffield__(self, item):
        unit = object.__getattribute__(self, "unit")
        if isinstance(unit, basestring):
            return unit
        try:
            i = object.__getattribute__(self,"dtype").names.index(item)
        except ValueError:
            raise ValueError("no field of name %s"%item)
        return unit[i]       



class Qrecarray(Qarray, np.recarray):
    
    def __getitem__(self, item):
        obj = np.recarray.__getitem__(self, item)        

        if isinstance(item, basestring):
            unit = self.__unitoffield__(item)
        else:
            unit = self.unit         
        return self.__qbuilder__(obj, unit)          
                  
    def __getattribute__(self, attr):
        # See if ndarray has this attr, and return it if so. (note that this
        # means a field with the same name as an ndarray attr cannot be
        # accessed by attribute).
        try:
            return object.__getattribute__(self, attr)
        except AttributeError:  # attr must be a fieldname
            pass
        return self[attr]        

    def __array_finalize__(self, obj):
        if issubclass(self.dtype.type, np.void):  
            if self.dtype.type is not Qrecord and self.dtype.fields:
                object.__setattr__(self, "dtype", np.dtype((Qrecord, self.dtype)))                

    def __unitoffield__(self, item):
        unit = object.__getattribute__(self, "_unit")
        if isinstance(unit, basestring):
            return unit
        try:
            i = object.__getattribute__(self,"dtype").names.index(item)
        except ValueError:
            raise ValueError("no field of name %s"%item)
        return unit[i] 

def _parse_void(v):
    if isinstance(v,np.dtype):
        return 
    raise TypeError()

def _parse_recarray(v):
    if isinstance(v,np.recarray):
        return 
    raise TypeError()    

###
# Record the array related calsses 
# WARNING order is important and ndarray must be at the
# end
QuantityTypes.__register_type__(np.recarray, Qrecarray)
QuantityTypes.__register_type__(np.record, Qrecord)
QuantityTypes.__register_type__(np.void, Qvoid)

### record the ndarray at the end
QuantityTypes.__register_type__(np.ndarray, Qarray, parser=np.asarray)


