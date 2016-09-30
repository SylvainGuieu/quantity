import math
import ast
import operator as op
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
NoneUnit = ''#None
NoneKind = ''#None
MixedSystem = 'mixed'
Hunitless = 1.0
UnitLess = "_"
KindUnitLess = "_"


eval_globals = {"math": math}

M_PYTHON, M_SCALE, M_NAME, M_PRINT, M_LATEX, M_HTML = range(6)
BASE_M = ['',1.0,'','','','']
K_NAME, K_PYTHON, K_DEFINITION, K_BASE, K_H, K_ALTH = range(6)

BASE_K = ['','','1.0','',1.0, []]
(UB_UNIT, UB_NAME, UB_PRINT, UB_LATEX, UB_HTML, UB_DEFINITION,
 UB_SYSTEM, UB_K) = range(8)

BASE_UB = ['', '', '', '', '', '', '', BASE_K]
U_BASE, U_METRIX, U_PYTHON, U_SCALE, U_DIMENSION = range(5)
BASE_U = [BASE_UB, BASE_M, '', 1.0, 1]

#(U_UNIT, U_NAME, U_PRINT, U_LATEX, U_HTML, U_DEFINITION, U_KIND, U_METRIX,
 #U_SCALE, U_SYSTEM, U_DIMENSION, U_ISMETRIX, U_H) = range(13)
C_IU, C_OU, C_PYTHON = range(3)


##################################################
#
# Parser Classes  
#
###################################################

#from .parser import conv, fconv

def hrounder(h):    
    return round(h,33)

class UnitError(NameError):
    pass    

class BadOpError(SyntaxError):
    pass

def _extract_nameerror(e):
    try:
        return str(e).split("name ")[1].split(" is ")[0]
    except:
        return e    

class RegisterError(ValueError):
    pass    


# unsupported operators
def _bnl(sign):
    def tmp(a,b):
        raise BadOpError("Operator %s not allowed in unit or kind description"%sign)
    return tmp        

operators = {
             ast.Add: _bnl("+"), 
             ast.Sub: _bnl("-"), 
             ast.BitXor: _bnl("~"), 
             ast.Or:  _bnl("|"), 
             ast.And: _bnl("|"), 
             ast.Mod: _bnl("%"),
             ast.Mult: op.mul,
             ast.Div: op.truediv, 
             ast.Pow: op.pow,
             ast.FloorDiv: op.truediv,              
             ast.USub: op.neg, 
             ast.UAdd: lambda a:a             
            }

class Parser(object):
    """ Parse a unit expression to its kind scale 

    e.g. :  "m/s"  -> 1.0
            "km/h" -> 3.6
    """
    def __init__(self, R):
        # save the register
        self.R = R 
        # The default operators
        self.operators = dict(operators)
        # Populate the operators with the one defined in this class
        self.init_op()
        self.buff = {}

    def init_op(self):
        """ Populate the operators with the one defined in this class
            They are matched by name    
        """    
        for a,f in [(ast.Num,"Num"), (ast.Mult,"Mult"), 
                    (ast.Div,"Div"), 
                    (ast.FloorDiv,"FloorDiv"), 
                    (ast.Pow, "Pow")                  
                  ]:
            if hasattr(self, f):
                self.operators[a] = getattr(self, f)        

    def Name(self, name):
        raise NotImplementedError("Name")            

    def Num(self, x):
        return x

    def wrap(self, output):
        return output    

    def eval(self, expr):
        try:
            r = self.Name(expr)
        except UnitError:            
            try:
                r = self.eval_(ast.parse(expr, mode='eval').body)
            except KeyError as e:
                raise UnitError("Unit %r is not registered"%e)
            #except TypeError as e:                
            #    raise UnitError("Unit expression is limited to simple mathematical expression, '%s' is not understood"%expr) 
            except BadOpError as e:
                raise BadOpError(e)
            else:
                return self.wrap(r)    
        else:
            return self.wrap(r)

    def eval_(self, node):    
        if isinstance(node, ast.Num): # <number>
            return self.Num(node.n)
        elif isinstance(node, ast.BinOp): # <left> <operator> <right>            
            return self.operators[type(node.op)](self.eval_(node.left), self.eval_(node.right))
        elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
            return self.operators[type(node.op)](self.eval_(node.operand))
        elif isinstance(node, ast.Attribute):
            return self.Attribute(node.value, node.attr) 

        elif isinstance(node, ast.Name):
            return self.Name(node.id) 

        else:
            raise TypeError(node)        
 
    def Attribute(self, left, right):
        """ allow only a attribute call to math, for e.i math.pi 
            however no function call is allowed like math.acos(-1)
        """        
        if not isinstance(left, ast.Name) or left.id!="math":
            raise TypeError("only math module allowed in unit definition")
        return getattr(math, right)

class ScaleParser(Parser):
    def Name(self, name):
        """ Name is the unit scale """
        try:
            return self.R.unit_lookup[name][U_SCALE]    
        except KeyError as e:
            raise UnitError("Unit %r is not registered"%name)

class MetrixScaleParser(Parser):
    def Name(self, name):
        """ Name is the unit scale """
        try:
            return self.R.metrix_lookup[name][M_SCALE]    
        except KeyError as e:
            raise UnitError("Metrix %r is not registered"%name)


class UnitHParser(Parser):
    def Name(self, name):
        try:
            return self.R.unit_lookup[name][U_BASE][U_K][K_H]    
        except KeyError as e:
            raise UnitError("Unit %r is not registered"%name)
            
class KindHParser(Parser):
    def Name(self, name):
        try:
            v = self.R.kind_lookup[name][K_H] 
        except KeyError:
            raise UnitError("Kind %r is not registered"%name)               
        else:
            return v            
    def Num(self, n):
        return n
        

GK_K, GK_D, GK_O = range(3)
class KindSpliter(Parser):

    def wrap(self,K):
        if not isinstance(K, list):
            if K!=1.0:
                raise KindError("%s not allowed in kind definition"%K)
            K = [(BASE_K, 1, "")]
        return SplitedKind(K)

                        
    def Name(self, name):
        try:
            K = self.R.kind_lookup[name] 
        except KeyError:
            raise UnitError("Kind %r is not registered"%name)               
        else:
            return [(K,1,"")]
    
    def Mult(self, left, right):
        if isinstance(left, list) and isinstance(right, list):
            for il, (LK,LDIM,LOP) in enumerate(left):
                for ir, (RK,RDIM,ROP) in enumerate(list(right)):
                    if LK is RK:
                        left[il] = (LK,LDIM+RDIM,LOP)
                        right.pop(ir)
            if right:
                RK,RDIM,ROP = right[0]
                right[0] = RK,RDIM,"*"                   
            return left+right
        
        if isinstance(right, list):
            if left!=1.0:
                raise BadOpError("In kind definition multiplication by constant is not allowed")                
            
            RK,RDIM,ROP = right[0]    
            right[0] = RK,RDIM,"*"
            return right

        if isinstance(left, list):
            if right!=1.0:
                raise BadOpError("In kind definition multiplication by constant is not allowed")                
            return left 
                
        return 1.0    

    def Div(self, left, right):
        if isinstance(left, list) and isinstance(right, list):
            for il, (LK,LDIM,LOP) in enumerate(left):
                for ir, (RK,RDIM,ROP) in enumerate(list(right)):
                    if LK is RK:
                        left[il] = (LK,LDIM-RDIM,LOP)
                        right.pop(ir)
                    #else:
                    #    right[ir] = (RK,-RDIM,"/")
            right = [(RK,-RDIM,"/") for (RK,RDIM,ROP) in right]                        
            return left+right
        

        if isinstance(right, list):
            if left!=1.0:
                raise BadOpError("In kind definition multiplication by constant is not allowed")                
            
            # reverse the dimension sign of all child
            return [(RK,-RDIM,"/") for (RK,RDIM,ROP) in right]

        if isinstance(left, list):
            if right!=1.0:
                raise BadOpError("In kind definition multiplication by constant is not allowed")                
            return left  
        return 1.0 

    FloorDiv = Div

    def Pow(self, left, right):
        if isinstance(right,list):
            raise BadOpError("kind in exponant isn't allowed")
        if isinstance(left, list):            
            return [(LK,LDIM*right,LOP) for LK,LDIM,LOP in left]                 
        if left**right != 1.0:
            raise BadOpError("In kind definition multiplication by constant is not allowed")            

class SplitedKind(object):
    def __init__(self, K):
        self.K = K

    @staticmethod
    def _hash(h,T):
        ## round problem when doing a negative exponant
        ## need to divide instead of **-1        
        if T[GK_D]<0:      
            return h/(T[GK_K][K_H]**-T[GK_D])
        return h*T[GK_K][K_H]**T[GK_D]

    def get_hash(self):
        return hrounder(reduce(self._hash, self.K, 1.0))

    @staticmethod
    def _althash(h,T):
        if T[GK_D]<0:      
            return h/(T[GK_K][K_ALTH][-1] if T[GK_K][K_ALTH] else T[0][K_H])**-T[1]
        return h*(T[GK_K][K_ALTH][-1] if T[GK_K][K_ALTH] else T[0][K_H])**T[1]
    def get_althash(self):
        return hrounder(reduce(self._althash, self.K, 1.0))
    
    @staticmethod
    def _name(name,T):
        d = -T[GK_D] if T[GK_O]=="/" else T[GK_D]
        o = "/" if T[GK_O]=="/" else ("." if T[GK_O]=="*" else "")        
        if d!=1:
            return "%s%s%s%s"%(name, o, T[GK_K][K_PYTHON], d)
        return "%s%s%s"%(name, o, T[GK_K][K_PYTHON])

    def get_name(self):
        return reduce(self._name, self.K, "")
    
    @staticmethod
    def _python(name,T):
        d = -T[GK_D] if T[GK_O]=="/" else T[GK_D]
        o = T[GK_O]
        if d!=1:
            return "%s%s%s**%s"%(name, o, T[GK_K][K_PYTHON], d)
        return "%s%s%s"%(name, o, T[GK_K][K_PYTHON])

    def get_python(self):
        return reduce(self._python, self.K, "")


    
    @staticmethod
    def _baseunit(name,T):
        d = -T[GK_D] if T[GK_O]=="/" else T[GK_D]
        o = T[GK_O]
        if d!=1:
            return "%s%s%s**%s"%(name, o, T[GK_K][K_BASE], d)
        return "%s%s%s"%(name, o, T[GK_K][K_BASE])    

    def get_baseunit(self):
        return reduce(self._baseunit, self.K, "")    

    def get(self, *args):
        return (getattr(self,"get_"+a)() for a in args)              




GU_U, GU_D, GU_O = range(3)
class UnitSpliter(Parser):
    def wrap(self,U):
        if not isinstance(U, list):
            U = [([BASE_UB, BASE_M, str(U), U, 1], 1, "")]
        return SplitedUnit(U)

    def Name(self, name):
        try:
            U = self.R.unit_lookup[name] 
        except KeyError:
            raise UnitError("Unit %r is not registered"%name)               
        else:
            # Unit definition, scale, dimention, prefered operator
            return [(U, 1,"")]
        
    def Mult(self, left, right):
        if not isinstance(left, list):
            left = [([BASE_UB, BASE_M, str(left), left, 1], 1, "")]
        if not isinstance(right, list):
            right = [([BASE_UB, BASE_M, str(right), right, 1], 1, "")]    

        
        for il, (LU, LDIM,LOP) in enumerate(left):
            for ir, (RU, RDIM,ROP) in enumerate(list(right)):
                if LU is RU:
                    left[il] = (LU, LDIM+RDIM,LOP)
                    right.pop(ir)
        if right:
            RU,RDIM,ROP = right[0]
            right[0] = RU,RDIM,"*"
        return left+right         

    def Div(self, left, right):
        if not isinstance(left, list):
            left = [([BASE_UB, BASE_M, str(left), left, 1], 1, "")]
        if not isinstance(right, list):
            right = [([BASE_UB, BASE_M, str(right), right, 1], 1, "")]    

        
        for il, (LU,LDIM,LOP) in enumerate(left):
            for ir, (RU,RDIM,ROP) in enumerate(list(right)):
                if LU is RU:
                    left[il] = (LU,LDIM-RDIM,LOP)
                    right.pop(ir)
                #else:
                #    right[ir] = (RK,-RDIM,"/")
        right = [(RU,-RDIM,"/") for (RK,RDIM,ROP) in right]                        
        return left+right        
    FloorDiv = Div

    def Pow(self, left, right):
        if isinstance(right,list):                        
            raise BadOpError("unit in exponant isn't allowed")
           
        if isinstance(left, list):            
            return [(LK,LDIM*right,LOP) for LK,LDIM,LOP in left]                 
        return left*right  




def _real_name(T, iu, im):
    U = T[GU_U][U_BASE]
    uname = U[iu] if U[iu] else T[GU_U][U_PYTHON]
           
    M = T[GU_U][U_METRIX]
    mname = M[im] if M[im] else M[M_PYTHON]
    return mname, uname
class SplitedUnit(object):
    def __init__(self, U):        
        self.U = U

    @staticmethod
    def _hash(h,T): 
        ## round problem when doing a negative exponant
        ## need to devide instead of **-1        
        if T[GU_D]<0:        
            return h/((T[GU_U][U_BASE][UB_K][K_H])**-T[GU_D])
        return h*(T[GU_U][U_BASE][UB_K][K_H])**T[GU_D]
    def get_hash(self):
        return hrounder(reduce(self._hash, self.U, 1.0))

    @staticmethod
    def _althash(h,T):
        alth = T[GU_U][U_BASE][UB_K][K_ALTH]
        if T[GU_D]<0:       
            return h/(alth[-1] if alth else T[GU_U][U_BASE][UB_K][K_H])**-T[GU_D]
        return h*(alth[-1] if alth else T[GU_U][U_BASE][UB_K][K_H])**T[GU_D]

    def get_althash(self):
        return hrounder(reduce(self._althash, self.U, 1.0))
                
    @staticmethod
    def _name(name,T):
        if T[GU_D]==0:
            return name        
        d = -T[GU_D] if T[GU_O]=="/" else T[GU_D]
        o = "per" if T[GU_D]<0 else ""
        
        mname, uname = _real_name(T, UB_NAME, M_NAME)
        if mname:
            rname = "%s%s"%(mname,uname)
        else:
            rname = uname

        if not rname:
            return name
        
        d *= T[GU_U][U_DIMENSION]

        if abs(d)==1:
            return "%s%s%s"%(rspacer(name), rspacer(o), rname)
        elif abs(d)==2:
            return "%s%ssquare %s"%(rspacer(name), rspacer(o), rname)
        elif abs(d)==3:
            return "%s%scubic %s"%(rspacer(name), rspacer(o), rname)
        else:    
            return "%s%s%s^%s"%(rspacer(name), rspacer(o), rname, abs(d))

        
    def get_name(self):
        return reduce(self._name, self.U, "")
    

    @staticmethod
    def _latex(name,T):
        if T[GU_D]==0:
            return name  

        d = -T[GU_D] if T[GU_O]=="/" else T[GU_D]
        o = "/" if T[GU_O]=="/" else ("." if T[GU_O]=="*" else "")        
        
        mname, uname = _real_name(T, UB_LATEX, M_LATEX)
        if mname:
            rname = "%s %s"%(mname,uname)
        else:
            rname = uname
            
        if not rname:
            return name

        d *= T[GU_U][U_DIMENSION]    

        if d!=1:
            return "%s%s%s^{%s}"%(rspacer(name), rspacer(o), rname, d)
        return "%s%s%s"%(rspacer(name), rspacer(o), rname)
        
    def get_latex(self):
        return reduce(self._latex, self.U, "")    

    @staticmethod
    def _print(name,T):
        if T[GU_D]==0:
            return name

        d = -T[GU_D] if T[GU_O]=="/" else T[GU_D]
        o = "/" if T[GU_O]=="/" else ("." if T[GU_O]=="*" else "")        
        
        mname, uname = _real_name(T, UB_PRINT, M_PRINT)
        if mname:
            rname = "%s%s"%(mname,uname)
        else:
            rname = uname
            
        if not rname or rname=="1.0":
            return name

        d *= T[GU_U][U_DIMENSION]

        if d!=1:
            return "%s%s%s%s"%((name), (o), rname, d)
        return "%s%s%s"%((name), (o), rname)
        
    def get_print(self):
        return reduce(self._print, self.U, "")     


    @staticmethod
    def _python(name,T):
        if T[GU_D]==0:
            return name
        d = -T[GU_D] if T[GU_O]=="/" else T[GU_D]
        o = T[GU_O]
        if d!=1:
            return "%s%s%s**%s"%(name, o, T[GU_U][U_PYTHON], d)
        return "%s%s%s"%(name, o, T[GU_U][U_PYTHON])

    def get_python(self):
        return reduce(self._python, self.U, "")

    @staticmethod
    def _scale(s, T):
        return s*(T[GU_U][U_SCALE])**T[GU_D]

    def get_scale(self):
        return reduce(self._scale, self.U, 1.0)    
    

    @staticmethod
    def _dimension(dims, T):
        kind = T[GU_U][U_BASE][UB_K][K_PYTHON]
        dims[kind] = dims.setdefault(kind,0) + T[GU_D]
        return dims

    def get_dimensions(self):
        return reduce(self._dimension, self.U, {})    
    

    @staticmethod
    def _system(sys, T):
        ns = T[GU_U][U_BASE][UB_SYSTEM]
        if ns and sys and ns!=sys:
            return MixedSystem
        else:
            return sys or ns 

    def get_system(self):
        return reduce(self._system, self.U, "")

    def get(self, *args):
        return (getattr(self,"get_"+a)() for a in args)              



def _decompose(R, U, depth=0):
    #U = R.unitparser.eval(unit).U
    if not depth:
        return U

    out = []
    found = False
    for T in U:
        deff = T[GU_U][U_BASE][UB_DEFINITION]
        mscale = T[GU_U][U_METRIX][M_SCALE]
        if deff and deff not in ["1.0", "1"]:
            found = True
            
            if T[GU_D]==1:
                if mscale!=1.0:
                    deff = "%s*%s"%(mscale,deff)
            else:
                if mscale!=1.0:
                    deff = "%s*(%s)**%s"%(mscale,deff,T[GU_D])        
                else:
                    deff = "(%s)**%s"%(deff,T[GU_D])    
                        
            new = _decompose(R, R.unitparser.eval(deff).U, depth-1)            
            if new:
                u,d,op = new[0]                        
                new[0] = u,d,T[GU_O]                
            out.extend(new)
        elif mscale!=1:
            deff = "%s*%s"%(mscale,T[GU_U][U_BASE][UB_UNIT])
            new = _decompose(R, R.unitparser.eval(deff).U, depth-1)            
            if new:
                u,d,op = new[0]                        
                new[0] = u,d,T[GU_O]                
            out.extend(new)                    
        else:
            out.append(T)            
    return _decompose(R, out, (depth-1)*found)

def decompose(R, unit, depth=0):
    U = R.unitparser.eval(unit).U
    # out = []
    # scale = 1.0
    # for T in U:
    #     if T[GU_U][U_BASE][UB_K][K_H]==1.0:
    #         scale *= T[GU_U][U_SCALE]
    #     else:
    #         out.append(T)

    # if scale!=1.0:    
    #     out = R.unitparser.eval("%s"%scale).U+out  
    # U = out            
    U = _decompose(R, U, depth)
    return R.unitparser.eval( SplitedUnit(U).get_python() )

def decomposition(R, unit):
    out = [unit]
    prev = ""
    new  = unit
    while prev!=new:
        prev = new
        new = decompose(R, prev, depth=1).get_python()
        if new!=prev:
            out.append(unitcode(R, new))
    return out

        
def group(R, su1, su2):
    pairs = []
    rest1 = []
    rest2 = []
    d1 = list(su1)
    d2 = list(su2)

    for i1, (U1, dim1, op1) in enumerate(list(d1)):
        K1 = U1[U_BASE][UB_K]
        K1A = K1[K_ALTH]
        for i2 ,(U2, dim2, op2) in enumerate(list(d2)):

            K2 = U2[U_BASE][UB_K]
            H2 = K2[K_H]
            H1 = K1[K_H]
            K2A = K2[K_ALTH]
            

            H1B = K1A[-1] if K1A else H1
            H2B = K2A[-1] if K2A else H2
            
            if H1B==H2B and H2 in K1A:                    
                T1 = H1
                convertors = []
                for T2 in K1A[:K1A.index(H2)+1]:
                    
                    try:
                        c = R.convertor_lookup[(T1,T2)]
                    except KeyError:
                        pass                            
                    else:                            
                        convertors.append(c)
                    T1=T2    
                if convertors:
                    pairs.append( [[U1,dim1,op1],
                                   [U2,dim2,op2],                                  
                                  _make_convertor(R, convertors)]
                                  )
                    d1.pop(i1)
                    d2.pop(i2)
                    continue   

            
            if H1B==H2B and H1 in K2A:                    
                T1 = H1
                convertors = []                    
                for T2 in (K2A[:K2A.index(T1)+1][::-1])+[H2]:
                    
                    try:
                        c = R.convertor_lookup[(T1,T2)]
                    except KeyError:
                        pass
                    else:
                        convertors.append(c)
                    T1 = T2    
                if convertors:

                    pairs.append([[U1,  dim1, op1],
                                  [U2,  dim2, op2],
                                  _make_convertor(R, convertors)])
                    d2.pop(i2) 
                    d1.pop(i1)   
                    continue 
    return SplitedUnit(d1), SplitedUnit(d2), pairs      

                                                                                          

                                                                            

def _make_convertor(R, convertors):
    def convertor(value, unit, outputunit):
        for iu,ou,c in convertors:
            if unit!=iu:
                value *= R.unit_lookup[unit][U_SCALE]/R.unit_lookup[iu][U_SCALE]                
            value = c(value)
            unit = ou
        if ou!=outputunit:
            value *= R.unit_lookup[ou][U_SCALE]/R.unit_lookup[outputunit][U_SCALE]            
        return value
    return convertor            

#############################################################
#
# Registery 
#
#############################################################



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
        # also organize the unit by kind/system
        self.scale2systemunit_lookup = {}


        # units
        self.unit_lookup = {}    
                
        # convertor
        self.convertor_lookup = {}
        self._init_parser()
        ## save some time to the interpretor build them 
        ## before
    def _init_parser(self):
        self.unitparser = UnitSpliter(self)
        self.kindparser = KindSpliter(self)
                
    

    def iterunits(self, kind=None):
        """ iterrator on unit string names """
        if kind:
            H = self.kindparser.eval(kind).get_hash()
            try:
                K = self.hash2kind_lookup[H]
            except KeyError:
                return
            else:        
                for u,i in self.unit_lookup.iteritems():            
                    if K is i[U_BASE][UB_K]: 
                        yield u     
        else:
            for u in self.unit_lookup:            
                yield u

    def iterkinds(self):
        """ iterator on kind string names """
        for k in self.kind_lookup:
            yield k         


def parentize(s):
    if isinstance(s, basestring) and any(o in s for o in "*/+-"):
        return "(%s)"%s
    return s

def texparentize(s):
    if isinstance(s, basestring) and any(o in s for o in "*/+-."):
        return "(%s)"%s
    return s    

def rspacer(s):
    if s:
        return s+" "
    return s    
def lspacer(s):
    if s:
        return " "+s
    return s    
def spacer(s):
    if s:
        return " "+s+" "
    return s
        
htmlparentize = texparentize

######################################################################################
#
#  Makers
#
######################################################################################



def make_metrix(R, metrix, scale, name=None, prt=None, latex=None, html=None, callback=None):
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
                   unicode(name)  if name else None, 
                   unicode(prt)   if prt else None, 
                   unicode(latex) if latex else None,
                   unicode(html)  if html else None
                ]   
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

def make_kind(R, kind, definition="", unitbase="", name=None):
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
    
    unitbase: string, optional
        The base unit of the kind (mostly the SI unit). e.g. is 'm' for 'length' kind

    name : string, optional
        A long name for the new kind.    
    

    """    
    if kind in R.kind_lookup:
        raise ValueError("kind '%s' already exists"%kind)

    name = name or kind

    if definition:
        H = R.kindparser.eval(definition).get_hash()
    else:
        H = hrounder(float(hash(kind)))

    if H in R.hash2kind_lookup:
        alt_Hs = [H]+R.hash2kind_lookup[H][K_ALTH]
        H = hrounder(float(hash(kind)))
    else:
        alt_Hs = []        
        #print("Warning '%s' kind shares the same hashing than '%s'."%(kind,R.hash2kind_lookup[H][K_PYTHON]))
                
    newkind = [name, kind, definition, unitbase, H, alt_Hs]        
    R.kind_lookup[kind] = newkind
    R.hash2kind_lookup[H] = newkind
    R.scale2unit_lookup[H] = {}
    R.scale2systemunit_lookup[H] = {}

        
def remove_kind(R, kind, callback=None):
    if R.__isglobal__:
        raise RegisterError("Cannot remove kind on global register")    

    K= R.kind_lookup[kind]
    H = K[K_H]    
    del R.hash2kind_lookup[H]
    del H
        
    del R.kind_lookup[kind]
    
    

    removed = []
    for unit, info in R.unit_lookup.items():
        if info[U_BASE][UB_K] is K:
            removed.append(remove_unit(R, unit, callback=callback))

    del R.scale2unit_lookup[H]
    del K    
    return removed        
                    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Units

def make_unit(R, unit, scale_or_definition, kind, dimension=1, metrix=False, name=None, 
                        prt=None, latex=None, html=None, system="", callback=None):
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

    latex : string, optional
       latex representation of unit
    
    html : string, optional
       html representation of unit

    """    
                
    #except:
    #    dimension = 1 

    if unit in R.unit_lookup and R.__isglobal__:
        raise ValueError("Unit '%s' already exists in the global unit register. You need to create your hown register"%unit)

    definition = scale_or_definition
    if isinstance(scale_or_definition, basestring):
        scale = R.unitparser.eval(scale_or_definition).get_scale()        
    else:
        scale = float(scale_or_definition)
        
    K = R.kind_lookup[kind]

    ubaseinfo = [
        unit, #UB_UNIT python name
        name, #UB_NAME nice name
        unicode(prt)  if prt else None,  # UB_PRINT print name 
        unicode(latex)if latex else None, 
        unicode(html) if html else None, 
        definition, 
        system, # UB_SYSTEM  system of the unit 
        K # UB_K kind definition of the unit     
    ]


    #if dimension>1:        
    #    scale = scale**dimension
        
    uinfo = [ubaseinfo,         
             BASE_M, 
             unit,
             scale, 
             dimension
            ]
    H = K[K_H]

    R.unit_lookup[unit] = uinfo
    R.scale2unit_lookup[H].setdefault(scale, uinfo)
    if system and system != MixedSystem:
        R.scale2systemunit_lookup[H].setdefault(system,{}).setdefault(scale, uinfo)
    
    if callback:      
        callback(unit)

    if metrix:
        for metrix_info in R.metrix_lookup.itervalues():
            make_metrix_unit(R, uinfo, metrix_info, callback)
                
def remove_unit(R, unit, callback=None):
    if R.__isglobal__:
        raise RegisterError("cannot remove unit on global register")
    i =  R.unit_lookup[unit]   
    H =  i[U_BASE][UB_K][K_H]
    scale = i[U_SCALE]

    del R.unit_lookup[unit]
    del R.scale2unit_lookup[H][scale]

    if callback:      
        callback(unit)
    return unit
        

def _make_m_mame(name1, alt1, name2, alt2, sep=""):
    if not name1 and not name2:
        return None
    elif name1 is None:
        return alt1+sep+name2
    elif name2 is None:
        return name1+sep+alt2
    return name1+sep+name2
        
def make_metrix_unit(R, unit, metrix, callback=None):
    metrix_info = R.metrix_lookup[metrix] if isinstance(metrix, basestring) else metrix
    unit_info   = R.unit_lookup[unit] if isinstance(unit, basestring) else unit

    metrix = metrix_info[M_PYTHON]
    unitbase = unit_info[U_BASE]
        
    metrix_scale = metrix_info[M_SCALE]
    dimension = unit_info[U_DIMENSION]

    if dimension>1:
        metrix_scale = metrix_scale**dimension

    metrix_scale *= unit_info[U_SCALE] # multiply by the unit scale
    
    newunitname = metrix_info[M_PYTHON]+unit_info[U_BASE][UB_UNIT]
    newinfo = [
               unit_info[U_BASE], 
               metrix_info, 
               newunitname,
               metrix_scale, 
               dimension
            ]                    
    H =  unit_info[U_BASE][UB_K][K_H]        

    R.unit_lookup[newunitname] = newinfo
    R.scale2unit_lookup[H].setdefault(metrix_scale, newinfo)
    system  = unit_info[U_BASE][UB_SYSTEM]

    if system and system != MixedSystem:
        R.scale2systemunit_lookup[H].setdefault(system,{}).setdefault(metrix_scale, newinfo)
        
    if callback:
        callback(metrix_unit)        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  Convertos



def make_convertor(R, kinds, targets, inputunit, outputunit, func):
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
    inputunit: string 
        the unit that should be passed into func
    outputunit: string 
        the unit returned by func   
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
            H =  "*" if  kind=="*"  else hashofkind(R, kind)
            Ht = "*" if target=="*" else hashofkind(R, target)
            if H==Ht:
                raise ValueError("building convertor, kinds are equal '%s'"%kind)
            convertor_lookup[(H, Ht)] = [inputunit, outputunit, convertor_func]
            #convertor_lookup[(H, Ht)] = convertor_func   
    return
    ######################################################################################
    #
    #  Define the API function.
    #
    ######################################################################################


def _applyconvertors(R, value, unit1, unit2, H, altHs, rev=False):
    unit = unit1
    for H2 in altHs:
        try:            
            if rev:
                uin, uout, convertor = R.convertor_lookup[(H,H2)]                
            else:    
                uin, uout, convertor = R.convertor_lookup[(H2,H)]
        except KeyError:
            unit = unit2
            value *= R.unit_lookup[unit1][U_SCALE]/R.unit_lookup[unit2][U_SCALE]
        else:                
            if uin!=unit:
                value *= R.unit_lookup[unit1][U_SCALE]/R.unit_lookup[uin][U_SCALE]

            value = convertor(value)
            unit = uout

    if unit2!=unit:
        value *= R.unit_lookup[uout][U_SCALE]/R.unit_lookup[unit2][U_SCALE]
        unit1 = uout
    return value



def _convert(R, value, unit1, unit2):
        
    us1 = R.unitparser.eval(unit1)
    us2 = R.unitparser.eval(unit2)
    h1, ht1, s1 = us1.get("hash", "althash", "scale")
    h2, ht2, s2 = us2.get("hash", "althash", "scale")
    

    if (h1)==(h2):
        return value * s1/s2
    if (ht1)!=(ht2):
        k1 = R.hash2kind_lookup[h1][K_PYTHON] if h1 in R.hash2kind_lookup else "unknown-kind"
        k2 = R.hash2kind_lookup[h2][K_PYTHON] if h2 in R.hash2kind_lookup else "unknown-kind"        
        raise UnitError("impossible conversion from a %r to a %r"%(k1,k2))        

    s1,s2,convertors = group(R, us1.U, us2.U);

    H1,Ht1,scale1 = s1.get("hash", "althash", "scale")
    H2,Ht2,scale2 = s2.get("hash", "althash", "scale")

    if Ht1!=Ht2:
        k1 = R.hash2kind_lookup[h1][K_PYTHON] if h1 in R.hash2kind_lookup else "unknown-kind"
        k2 = R.hash2kind_lookup[h2][K_PYTHON] if h2 in R.hash2kind_lookup else "unknown-kind"
        raise UnitError("Impossible conversion from %r to %r"%(k1,k2))

    value *= scale1/scale2
    for d1,d2,c in convertors:
        if d1[GU_D]!=d2[GU_D]:
            raise UnitError("Dimension mismatch")
        value = c(value, d1[GU_U][U_PYTHON], d2[GU_U][U_PYTHON] )**d1[GU_D]
    return value


def convert(R, value, unit1, unit2=None, system=None, inside=lambda x,u:x):
    if unit2 is None:
        if system is None:
            unit2 = baseofunit(R, untit1)
        else:
            H = R.unitparser.eval(unit1).get_hash()
            try:    
                sys_lookup = R.scale2systemunit_lookup[H]
            except KeyError:
                raise KindError("No recorded system for this kind")    
            else:
                try:
                    scale_lookup = sys_lookup[system]
                except KeyError:
                    raise KindError("No system found for this kind")
                else:
                    uscale = scaleofunit(R, unit1)
                    keys = scale_lookup.keys()
                    ##
                    ## One could take the scale the closest to the value, but sometime it ends up
                    ## to weird units like yoctoparsec ! 
                    #diff, i = min((abs(value*uscale-s), idx) for (idx, s) in enumerate(keys))
                    ##
                    ## Or maybe the lower scale of the unit ?
                    # diff, i = min((abs(uscale), idx) for (idx, s) in enumerate(keys))
                    # unit2 = scale_lookup[keys[i]][U_PYTHON]
                    ## better to take the base unit of the system                     
                    for scale, U in scale_lookup.iteritems():
                        if U[U_PYTHON] == U[U_BASE][UB_K][K_BASE]:
                            unit2 = scale_lookup[scale][U_PYTHON]
                            break
                    else:
                        diff, i = min((abs(uscale), idx) for (idx, s) in enumerate(keys))
                        unit2 = scale_lookup[keys[i]][U_PYTHON]        

    elif system is not None:
        raise ValueError("convertion unit and system cannot be both given")
    
    return inside(_convert(R, value, unit1, unit2), unit2)


def _bestscale(scales, scale):
    dif = [abs(s-scale) for s in scales]
    return scales[dif.index(min(dif))]

def convertsystem(R, value, unit, system):
    ## try the fast way 
    H, Ht, usystem, uscale = R.unitparser.eval(unit).get("hash", "althash", "system", "scale")

    try:
        scale_lookup = R.scale2systemunit_lookup[H]
    except KeyError:
        pass
    else:
        try:
            scale2unit = scale_lookup[system]
        except KeyError:
            raise NameError("system %r is unknown for that kind "%(system))    
        
        bestscale = _bestscale(scale2unit.keys(), uscale)
        return value*uscale/bestscale, scale2unit[bestscale][U_PYTHON]
    

def kindofkind(R, kind):
    H, Ht = R.kindparser.eval(kind).get("hash", "althash")
    try:
        i = R.hash2kind_lookup[H]
    except KindError:
        try:
            i = R.hash2kind_lookup[Ht]
        except KindError:
            return kind                    
    return i[K_PYTHON]


def kindofunit(R, unit):
    """ return the kind string of a unit or None if unit does not exist"""

    # try the fast way     
    H, Ht, name = R.unitparser.eval(unit).get("hash", "althash", "python")
    try:
        return R.hash2kind_lookup[H][K_PYTHON]
    except KeyError:
        try:
            return R.hash2kind_lookup[Ht][K_PYTHON]        
        except KeyError:
            return None

def systemofunit(R, unit):
    return R.unitparser.eval(unit).get_system()

def scaleofunit(R, unit):
    """ return the scale factor of a unit """
    return R.unitparser.eval(unit).get_scale() if unit else 1.0    

def scaleofmetrix(R, metrix):
    """ return the scale factor of a metrix """
    try:
        i = R.metrix_lookup[metrix]
    except KeyError:
        return None
    return i[M_SCALE]        

def latexofmetrix(R, metrix):
    """ return the scale factor of a metrix """
    try:
        i = R.metrix_lookup[metrix]
    except KeyError:
        return None
    return i[M_LATEX]

def htmlofmetrix(R, metrix):
    """ return the scale factor of a metrix """
    try:
        i = R.metrix_lookup[metrix]
    except KeyError:
        return None
    return i[M_HTML]

def nameofmetrix(R, metrix):
    """ return the scale factor of a metrix """
    try:
        i = R.metrix_lookup[metrix]
    except KeyError:
        return None
    return i[M_NAME]


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
    if not unit:
        return NoneUnit, NoneKind

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
    H = R.kindparser.eval(kind).get_hash()
    try:
        i =  R.scale2unit_lookup[H][float(scale)]
    except KeyError:
        return None
    return i[U_PYTHON]

def hashofkind(R, kind):    
    return R.kindparser.eval(kind).get_hash()

def hashofunit(R, unit):
    return R.unitparser.eval(unit).get_hash()

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

    return getattr(value, "value", value) 
    
    Parameter
    ---------
    value : numerical like, quantity

    Outputs
    -------
    value : numerical
    """
    return getattr(value, "value", value)

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
        H,Ht = R.kindparser.eval(kind).get("hash", "althash")
        return Ht==Hunitless

    return (kind is None and unitof(value)=='')

               
def nameofkind(R, kind):
    """ return the  string name of a kind or None
    
    Parameters
    ----------
    kind : string
    
    Outputs
    -------
    name : string or None
    
    Examples
    --------
    >>> kindofunit( "km/h" )     
    u'velocity'
    """
    return R.kindparser.eval(kind).get_name()


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
    try:
        K = R.kind_lookup[kind]
    except KeyError:
        return []                
    return [k for k,i in R.unit_lookup.iteritems() if i[U_BASE][UB_K] is K]

def _printisize(s):
    return s

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
    return R.unitparser.eval(unit).get_print()



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
    if not unit:
        return NoneUnit

    return R.unitparser.eval(unit).get_name()

def _latexize(u):
    return u
def latexofunit(R, unit):
    """ return the  string latex representation of a unit or None
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    latex : string or None
        
    """
    if not unit:
        return NoneUnit
    return R.unitparser.eval(unit).get_latex()        

def _htmlize(u):
    return u

def htmlofunit(R, unit):
    """ return the  string html representation of a unit or None
    
    Parameters
    ----------
    unit : string
    
    Outputs
    -------
    latex : string or None
        
    """
    if not unit:
        return NoneUnit
    return R.unitparser.eval(unit).get_html()     

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
    if not unit:
        return NoneUnit

    info = _getunitinfo(R, unit)
    return unicode(info[U_METRIX][U_NAME]) if info else None

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
    if not unit:
        return NoneUnit
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
    return R.unitparser.eval(unit).get_dimensions()


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
    if not kind:
        return NoneUnit
    return R.kindparser.eval(kind).get_baseunit()

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
    return R.unitparser.eval(unit).get_hash() == R.kindparser.eval(kind).get_hash()

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

    H = hashofkind(R, kind) if isinstance(kind, basestring) else kind
    Ht = hashofkind(R, kind_targeted) if isinstance(kind_targeted, basestring) else kind_targeted
    try:
        c = R.convertor_lookup[(H,Ht)]
    except KeyError:
        try:
            c = R.convertor_lookup[("*", Ht)]
        except KeyError:
            try:
                c = R.convertor_lookup[(H, "*")]
            except:
                c = None
    return c[C_PYTHON]

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
    H = hashofkind(R, kind) if isinstance(kind, basestring) else kind
    for cH, target in R.convertor_lookup.keys():
        if H == cH:
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


class unitcode(unicode):
    def __new__(cl, R, u):
        self = unicode.__new__(cl, unicode(u).strip())
        self.R = R
        return self
                    
    @property
    def kind(self):
        return kindcode(self.R, kindofunit(self.R, self))
    
    @property
    def name(self):
        return nameofunit(self.R, self)        

    @property
    def prt(self):
        return printofunit(self.R, self)   

    @property
    def decompose(self):
        return decomposeunit(self.R, self)
         
    @property
    def pow(self):
        return unitcode(self.R, unitpowofunit(self.R, self))
            
    @property
    def latex(self):
        return latexofunit(self.R, self)        

    @property
    def html(self):
        return htmlofunit(self.R, self) 
    
    @property
    def scale(self):
        return scaleofunit(self.R, self) 

    @property
    def base(self):
        return baseofunit(self.R, self)
    
    @property
    def metrix(self):
        return metrixcode(self.R, metrixofunit(self.R, self))
    
    @property
    def dimensions(self):
        return dimensionsofunit(self.R, self)

class metrixcode(unicode):
    def __new__(cl, R, u):
        self = unicode.__new__(cl,unicode(u).strip())
        self.R = R
        return self
    
    @property
    def scale(self):
        return scaleofmetrix(self.R,self)
    
    @property
    def name(self):
        return nameofmetrix(self.R, self)        

    @property
    def latex(self):
        return latexofmetrix(self.R, self)        

    @property
    def html(self):
        return htmlofmetrix(self.R, self) 

class kindcode(unicode):
    def __new__(cl, R, u):
        self = unicode.__new__(cl,unicode(u).strip())
        self.R = R
        return self

    @property
    def units(self):
        return [unitcode(self.R, u) for u in self.R.iterunits(self)]

    @property
    def name(self):
        return nameofkind(self.R, self)    

    @property
    def base(self):
        return unitcode(self.R, baseofkind(self.R, self))

class unitlist(list):
    def __init__(self,R, iterable):
        list.__init__(self, (unitcode(R,u) for u in iterable))
        self.R = R

    def __repr__(self):
        return ", ".join(self)    

    def append(self, obj):
        list.append(self, unitcode(self.R, obj))    
    
    def extend(self, obj):
        list.extend(self, (unitcode(self.R, u) for u in obj))    
    
    def __add__(self, obj):
        return list.__add__(self, unitlist(self.R, obj))
    
    def __radd__(self, obj):
        return list.__radd__(self, unitlist(self.R, obj))    

    def __mul__(self, num):
        return unitlist(self.R, list.__mul__(self, num))    
    
    def __rmul__(self, num):
        return unitlist(self.R, list.__rmul__(self, num))


    
