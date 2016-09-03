""" Populate the kind's namespace from the parsed tables """
from .api import baseofkind
class _KIND_:
    def __init__(self, name):
        self.kindname = name

    def __getitem__(self, unit):
        return self.__dict__[unit]
    
    def __iter__(self):
        return self.__dict__.iterkeys()

    @property
    def baseunit(self):
        return baseofkind(self.kindname)                