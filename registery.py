from __future__ import print_function
import weakref
from . import api


def func_name(f):
    try:
        return f.func_name # python 2
    except AttributeError:
        return f.__name__

def im_func(o):
    try:
        return o.im_func
    except AttributeError:
        return o.__func__

def redoc(f):
    f.__doc__ = getattr(api, func_name(f)).__doc__
    return f

def _get_class_attr(cl, attr):
    """ get a attribute with avoiding the __get__ method """
    for sub in cl.__mro__:
        for key,val in sub.__dict__.items():
            if key == attr:
                return val
    raise AttributeError("type object '%s' has no attribute '%s'"%(cl.__name__, attr)) 


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

class QuantityTypes(object):
    ## these vars will be filled up in base
    registered_quantity = []
    registered_quantity_parser = []    

    def __init__(self, R):
        if not R.__isglobal__:      
            registered_quantity = [] 
            registered_quantity_parser = []

            first = True
            for subcl, cl in self.registered_quantity:

                if first:
                    ## reinstance the convertor
                    tocl = _get_class_attr(cl, "to").__class__
                    toinstance = tocl.Instance
                    toinstance = type(toinstance.__name__, (toinstance.__mro__[1],), {})
                    tocl = type(tocl.__name__, (tocl,), {"Instance":toinstance})
                    first = False
                d = {
                     "R": R, 
                     "_QTwr_":weakref.ref(self), 
                     "to":tocl()
                    }    
                newcl =  type(cl.__name__, (cl,), d)
                registered_quantity.append((subcl, newcl))
                for pcl, parser in self.registered_quantity_parser:
                    if pcl==cl:
                        registered_quantity_parser.append((newcl, parser))  
                                  
            self.registered_quantity = registered_quantity
            self.registered_quantity_parser = registered_quantity_parser                    
        self.R = R
        self.__update__()

    def __update__(self):    
        ## Populate the attribute 
        for subcl, cl in self.registered_quantity:
            name = cl.__name__
            if name=="Unit":
                attr = "units"
            else:    
                attr = name.lstrip("Q")+"units"

            setattr(self, attr, cl.to)
            setattr(self, cl.__name__, cl)

            toinstance = _get_class_attr(cl, "to").Instance
            for u in self.R.iterunits():
                #if not hasattr(toinstance, u):
                    setattr(toinstance, u, UnitConverterProperty(u))
            #im_func(toinstance.__init__).__doc__ = ""

    def __updateunit__(self, unit):
        if self.registered_quantity:
            _, cl = self.registered_quantity[0]
            toinstance = _get_class_attr(cl, "to").Instance
            if unit in self.R.unit_lookup:
                setattr(toinstance, unit, UnitConverterProperty(unit))
            else:
                delattr(toinstance, unit)

    @classmethod     
    def __register_type__(me, subclass, cl, before=None, parser=None, replace=None):
        if before is not None and replace is not None:
            raise ValueError("before and replace cannot be both set")
        if before:
            for i, (s,c) in enumerate(me.registered_quantity):
                if before is s:
                    break                            
        elif replace:
            for i, (s,c) in enumerate(me.registered_quantity):
                if replace is s:
                    break                     
        else:
            i = len(me.registered_quantity)
                
        if replace: 
            me.registered_quantity[i] = (subclass, cl)
        else:                
            me.registered_quantity.insert(i, (subclass, cl))
        
        if parser:
            me.registered_quantity_parser.append((parser,cl))    
    
    def register_system_units(self, *args):        
        from .data_parser import  parse_metrix_table, parse_convertors_table, parse_kinds_table, parse_units_table
        from .data_quantity import (unit_txt_tables , kind_txt_tables, metrix_txt_tables,  convertor_txt_tables)
        args = list(args)

        if args:
            options = []
            for o in ["metrix","kinds","units","convertors"]:
                try:
                    options.append(args.remove())
                except:
                    continue
            if len(args):
                raise ValueError("Unknown option %s"%args) 
        else:
            options = ["metrix","kinds","units","convertors"]                       
        
        R = self.R

        if "metrix" in options:
            for tbl in metrix_txt_tables:
                parse_metrix_table(R, tbl)    

        if "kinds" in options:        
            for tbl in kind_txt_tables:
                parse_kinds_table(R, tbl)

        if "convertors" in options:        
            for tbl in convertor_txt_tables:
                parse_convertors_table(R, tbl)                

        if "units" in options:        
            for tbl in unit_txt_tables:
                parse_units_table(R, tbl)  

        self.__update__()          

            
    def quantitytype(self, tpe):
        for subcl, cl in self.registered_quantity:
            if issubclass(tpe, subcl):
                return cl
        raise TypeError("type %r has no equivalent for quantity"%tpe)

    def get_quantity_class(self, value):
        for subcl, cl in self.registered_quantity:
            if isinstance(value, subcl):
                return cl, value
        for parser, cl in self.registered_quantity_parser:
            try:
                v = parser(value)
            except:
                continue
            else:
                return cl, v        

        return None, value
    
    def parsevalue(self, value):
        for subcl, cl in self.registered_quantity:
            if isinstance(value, subcl):
                return value         
        for parser, cl in self.registered_quantity_parser:
            try:
                v = parser(value)
            except:
                continue
            else:
                return v
        raise ValueError("type '%s' is an invalid type for quantity")        

    @redoc    
    def make_metrix(self, metrix, scale, name=None, prt=None, latex=None, html=None):         
        return  api.make_metrix(self.R, metrix, scale, name=name, prt=prt, latex=latex, html=html, callback=self.__updateunit__)
           
    @redoc    
    def remove_metrix(self, metrix):
        return  api.remove_metrix(self.R, metrix, callback=self.__updateunit__)

    @redoc    
    def make_kind(self, kind, definition="", unitbase="", name=None, unitless=False):
        out =  api.make_kind(self.R, kind, definition=definition, unitbase=unitbase, name=name, unitless=unitless)
        self.__update__()
        return out

    @redoc    
    def remove_kind(self, kind):
        return  api.remove_kind(self.R, kind, callback=self.__updateunit__)
    
    @redoc
    def make_unit(self, unit, scale_or_definition, kind=None, dimension=1, metrix=False, name=None, prt=None, system=""):
        if isinstance(scale_or_definition, basestring):
            scale = self.scaleofunit(scale_or_definition)
            ukind = self.kindofunit(scale_or_definition)            
        else:
            if hasattr(scale_or_definition, "unit"):
                try:                
                    scale = float(scale_or_definition)
                except TypeError:
                    raise ValueError("unit quantity must be a scalar convertible to float")

                scale *= self.scaleofunit(scale_or_definition.unit)        
                ukind = self.kindofunit(scale_or_definition.unit)
            else:
                scale = scale_or_definition
                if kind is None:
                    raise ValueError("kind name is expected when unit not defined by a string or quantity")

        if kind is None:                            
            if not self.kindexist(ukind):
                raise ValueError("kind '%s' is not registered, provide a kind name to create one on the fly"%ukind)
            kind = ukind                
        else:            
            if self.kindexist(ukind) and ukind!=kind:
                raise ValueError("the kind already exists under the name '%s' cannot change to '%s'"%(ukind,kind))
            self.make_kind(kind, ukind)    

        return api.make_unit(self.R, unit, scale, kind,  dimension=dimension, metrix=metrix, name=name, prt=prt, system=system, callback=self.__updateunit__)                            
        
    @redoc    
    def remove_unit(self, unit):
        return  api.remove_unit(self.R, unit, callback=self.__updateunit__)

    @redoc    
    def make_convertor(self, kinds, targets, func):
        out = api.make_convertor(self.R, kinds, targets, func)    
        self.__update__()
        return out

    @redoc
    def convert(self, value, unit, newunit, inside=lambda x,u:x):
        return api.convert(self.R, value, unit, newunit, inside=inside)

    @redoc
    def kindofunit(self, unit):    
        return api.kindofunit(self.R, unit)

    @redoc
    def scaleofunit(self, unit):
        return api.scaleofunit(self.R, unit)
    
    @redoc
    def kindofkind(self, kind):
        return api.kindofkind(self.R, kind)

    @redoc
    def unitofunit(self, unit,kind=None):
        return api.unitofunit(self.R, unit,kind=kind)    

    @redoc    
    def unitofscale(self, scale, kind):
        return api.unitofscale(self.R, scale, kind)

    # @redoc    
    # def hashofkind(self, kind):
    #     return api.hashofkind(self.R, kind)

    # @redoc    
    # def hashofunit(R, unit):
    #     H =  eval(unit, eval_globals, R.kindofunithash_lookup)
    #     return H

    @redoc
    def _getkindinfo(R, kind):
        return R.kind_lookup.get(kind,None)

    @redoc
    def _getunitinfo(R, unit):
        unit, kind = unitofunit(R, unit)
        return R.unit_lookup.get(unit, None)


    @redoc
    def unitof(self, value):
        return api.unitof(value)

    @redoc
    def kindof(self, value):
        return api.kindof(value)        

    @redoc
    def valueof(self, value):
        return api.valueof(value)

    @redoc
    def scaleof(self, value):
        return api.scaleof(self.R, value)

    @redoc
    def isunitless(self, value):
        return api.isunitless(value)

    @redoc
    def unitsofkind(self, kind):
        return api.unitsofkind(self.R, kind)

    @redoc        
    def printofunit(self, unit):
        return api.unitsofkind(self.R, unit)

    @redoc        
    def definitionofunit(self, unit):
        return api.definitionofunit(self.R, unit)

    @redoc        
    def nameofunit(self, unit):
        return api.nameofunit(self.R, unit)

    @redoc        
    def metrixofunit(self, unit):
        return api.metrixofunit(self.R, unit)

    @redoc       
    def baseofunit(self, unit):
        return api.baseofunit(self.R, unit)

    @redoc        
    def basescaleofunit(self, unit):
        return api.basescaleofunit(self.R, unit)

    @redoc        
    def dimensionsofunit(self, unit):
        return api.dimensionsofunit(self.R, unit)

    @redoc      
    def baseofkind(self, kind):
        return api.baseofkind(self.R, kind)

    @redoc        
    def isunitof(self, unit, kind):
        return api.isunitof(self.R, unit, kind)

    @redoc
    def getconvertor(self, kind, kind_targeted):
        return api.getconvertor(self.R, kind, kind_targeted)

    @redoc       
    def linksofkind(self, kind):
        return api.linksofkind(self.R, kind)

    @redoc        
    def arekindconvertible(self, kind_from, kind_to):
        return api.arekindconvertible(self.R, kind_from, kind_to)

    @redoc        
    def areunitconvertible(self, unit_from, unit_to):
        return api.areunitconvertible(self.R, unit_from, unit_to)

    @redoc       
    def getkinds(self):
        return api.getkinds(self.R)

    @redoc
    def getunits(self):
        return api.getunits(self.R)        

    @redoc
    def unitexist(self, unit):
        return api.unitexist(self.R, unit)
    
    @redoc
    def kindexist(self, kind):
        return api.kindexist(self.R, kind)

    @redoc
    def metrixexist(self, metrix):
        return api.metrixexist(self.R, metrix)



