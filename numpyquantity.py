from . import base
from .base import (_quantity_shared,                     
                    _prepare_quantity_class, 
                    parseunit, 
                    QuantityTypes
                  )
from .base import Unit as BaseUnit
import numpy as np

__all__ = ["Qarray", "Qfloat128", "Qfloat64", "Qfloat32", "Qfloat16", 
           "Qint64", "Qint32", "Qint16", "Qint8", 
           "Quint64", "Quint32", "Quint16", "Quint8",
           "Qcomplex256", "Qcomplex128", "Qcomplex64", 
           "Unit"
        ]

def isarray(v):
    """ to check if an array of scalar """
    return hasattr(v, "__iter__")

class Unit(BaseUnit, np.ndarray):
    def __new__(cl, unit):
        new = np.asarray(1.0).view(cl)
        new._unit = unit
        new.__init_unit__()
        return new
    @property
    def unit(self):
        return self._unit

    def __array_wrap__(self, o):
        return self.__class__(self.unit)    

class Qarray(_quantity_shared, np.ndarray):
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

                
        self._unit = unit
        self._oview = oview

        if self.dtype.type is not Qvoid and self.dtype.fields:
            self.dtype = np.dtype((Qvoid, self.dtype))
            self.dtype.type.unit = self.unit      

        self.__init_unit__()    
        return self

    def __array_finalize__(self, obj):
        pass
        #if self.dtype.type is not Qvoid and self.dtype.fields:
        #    self.dtype = np.dtype((Qvoid, self.dtype))
        #    self.dtype.type.unit = self.unit      

    def __getitem__(self, item):
        sub = np.ndarray.__getitem__(self, item)
        if self._sameunit:
            unit = self.unit
        else:
            if isinstance(item, basestring):
                try:
                    i = self.dtype.names.index(item)
                except ValueError:
                    raise ValueError("no field of name %s"%item)
                unit = self.unit[i]
            else:
                unit = self.unit             
        try:
            len(sub)
        except TypeError:
            return self.QT().get_quantity_class(type(sub))[0](sub, unit)
            #return self.__qbuilder__(sub, unit)            
        else:
            return self.__qbuilder__(np.asarray(sub), unit)
                
    def __getslice__(self, a,b): 
        sub = np.ndarray.__getslice__(self,a,b)       
        return Qarray(np.asarray(sub), self.unit)
    
    def __tovalue__(self, v):
        return np.asarray(v, dtype=self.dtype)

    @property        
    def __qwrapper__(self):
        return lambda a,u: Qarray(np.asanyarray(a, self.dtype), u) 

    def __array_wrap__(self, o):
        return self.__class__(o, self.unit)

class Qfloat128(np.float128):
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.float128), unit)
        else:    
            new = np.float128.__new__(cl, v)
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()        
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()                
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()                 
        return new

    @staticmethod
    def __tovalue__(v):
        return np.float32(v)

    def __repr__(self):
        #print "AAAA"
        return "%r [%s]"%(self.__tovalue__(self), self.unit)    
                    
QuantityTypes.__register_type__(np.float32, Qfloat32)
_prepare_quantity_class(Qfloat32)


class Qfloat16(np.float16):        
    def __new__(cl, v, unit=""):
        if isarray(v):
            new = Qarray(np.asarray(v,dtype=np.float16), unit)
        else: 
            new = np.float16.__new__(cl, v)
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
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
            new._unit = parseunit(new.R, unit)
        new.__init_unit__()            
        return new

    @staticmethod
    def __tovalue__(v):
        return np.complex64(v)
QuantityTypes.__register_type__(np.complex64, Qcomplex64)
_prepare_quantity_class(Qcomplex64)

#value2type_lookup.insert(0, (np.float64, np.float64, Float64) ) 
#base.quantityclasses.append(Float64)


class Qvoid(_quantity_shared, np.void):        
    @staticmethod
    def __tovalue__(v):
        return np.dtype((np.void, v))

class Qrecarray(np.recarray, _quantity_shared):    
    def __new__(subtype, data_array, unit):
        obj = np.asanyarray(data_array)
        oview = type(obj)        
        obj = obj.view(subtype)

        if isinstance(unit, basestring):
            obj._sameunit = True
        else:
            obj._sameunit = False
            if len(unit) != len(obj.dtype):
                raise ValueError("Number of array field is %d number of units is %s"%( len(obj.dtype),len(unit)))

        obj._unit = unit        
        obj._oview = oview
        new.__init_unit__()
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        #self.unit = getattr(obj, 'unit', "")
        #self._ref_classes = (Qarray,)

    def __getitem__(self, item):
        sub = np.ndarray.__getitem__(self, item)
        if self._sameunit:
            unit = self.unit
        else:
            if isinstance(item, basestring):
                try:
                    i = self.dtype.names.index(item)
                except ValueError:
                    raise ValueError("no field of name %s"%item)
                unit = self.unit[i]
            else:
                unit = self.unit             
        try:
            len(sub)
        except TypeError:
            return self.__qbuilder__(sub, unit)            
        else:
            return self.__qbuilder__(np.asarray(sub), unit)

            
    def __getslice__(self, a,b): 
        sub = np.ndarray.__getslice__(self,a,b)       
        return self.__qbuilder__(np.asarray(sub), self.unit)

    # def __repr__(self):        
    #     return self._oview.__repr__(self)
    #     return np.asarray(self).__repr__()[0:-1]+", unit=%r)"%(self.unit)

    @staticmethod
    def __tovalue__(v):
        return np.asarray(v)
    ## 
    # not sure why but the operation as to 
    # be copied here 
    __mul__ = _quantity_shared.__mul__
    __rmul__= _quantity_shared.__rmul__
    __pow__ = _quantity_shared.__pow__
    __add__ = _quantity_shared.__add__ 
    __radd__= _quantity_shared.__radd__
    __neg__ = _quantity_shared.__neg__
    __pos__ = _quantity_shared.__pos__
    #__mod__ = _quantity_shared.__mod__
    #__rmod__= _quantity_shared.__rmod__
    __sub__ = _quantity_shared.__sub__
    __rsub__= _quantity_shared.__rsub__   
    __div__ = _quantity_shared.__div__
    __floordiv__ = _quantity_shared.__floordiv__
    __rdiv__ = _quantity_shared.__rdiv__
    __rfloordiv__ = _quantity_shared.__rfloordiv__



def _parse_void(v):
    if isinstance(v,np.dtype):
        return 
    raise TypeError()

def _parse_recarray(v):
    if isinstance(v,np.recarray):
        return 
    raise TypeError()    

QuantityTypes.__register_type__(np.void, Qvoid)
QuantityTypes.__register_type__(np.recarray, Qrecarray)
QuantityTypes.__register_type__(np.ndarray, Qarray, parser=np.asarray)

# value2type_lookup.insert(0, (np.void, _parse_void, Qvoid) ) 
# base.quantityclasses.append(Qvoid)

# value2type_lookup.insert(0, (np.recarray, _parse_recarray, Qrecarray) ) 
# base.quantityclasses.insert(0, Qrecarray)

# value2type_lookup.insert(0, (np.ndarray, np.asarray, Qarray) ) 
# base.quantityclasses.insert(0, Qarray)





