# This program is in the public domain
"""`igor.py` compatibility layer on top of the `igor` package.

igor.load('filename') or igor.loads('data') loads the content of an igore file
into memory as a folder structure.

Returns the root folder.

Folders have name, path and children.
Children can be indexed by folder[i] or by folder['name'].
To see the whole tree, use: print folder.format()

The usual igor folder types are given in the technical reports
PTN003.ifn and TN003.ifn.
"""
from __future__ import absolute_import
import io as _io
import locale as _locale
import re as _re
import sys as _sys

import numpy as _numpy

from .binarywave import MAXDIMS as _MAXDIMS
from .packed import load as _load
from .record.base import UnknownRecord as _UnknownRecord
from .record.folder import FolderStartRecord as _FolderStartRecord
from .record.folder import FolderEndRecord as _FolderEndRecord
from .record.history import HistoryRecord as _HistoryRecord
from .record.history import GetHistoryRecord as _GetHistoryRecord
from .record.history import RecreationRecord as _RecreationRecord
from .record.packedfile import PackedFileRecord as _PackedFileRecord
from .record.procedure import ProcedureRecord as _ProcedureRecord
from .record.wave import WaveRecord as _WaveRecord
from .record.variables import VariablesRecord as _VariablesRecord


__version__='0.10'


ENCODING = _locale.getpreferredencoding() or _sys.getdefaultencoding()
PYKEYWORDS = set(('and','as','assert','break','class','continue',
                  'def','elif','else','except','exec','finally',
                  'for','global','if','import','in','is','lambda',
                  'or','pass','print','raise','return','try','with',
                  'yield'))
PYID = _re.compile(r"^[^\d\W]\w*$", _re.UNICODE)
def valid_identifier(s):
    """Check if a name is a valid identifier"""
    return PYID.match(s) and s not in PYKEYWORDS


class IgorObject(object):
    """ Parent class for all objects the parser can return """
    pass

class Variables(IgorObject):
    """
    Contains system numeric variables (e.g., K0) and user numeric and string variables.
    """
    def __init__(self, record):
        self.sysvar = record.variables['variables']['sysVars']
        self.uservar = record.variables['variables']['userVars']
        self.userstr = record.variables['variables']['userStrs']
        self.depvar = record.variables['variables'].get('dependentVars', {})
        self.depstr = record.variables['variables'].get('dependentStrs', {})

    def format(self, indent=0):
        return " "*indent+"<Variables: system %d, user %d, dependent %s>"\
            %(len(self.sysvar),
              len(self.uservar)+len(self.userstr),
              len(self.depvar)+len(self.depstr))

class History(IgorObject):
    """
    Contains the experiment's history as plain text.
    """
    def __init__(self, data):
        self.data = data
    def format(self, indent=0):
        return " "*indent+"<History>"

class Wave(IgorObject):
    """
    Contains the data for a wave
    """
    def __init__(self, record):
        d = record.wave['wave']
        self.name = d['wave_header']['bname'].decode(ENCODING)
        self.data = d['wData']
        self.fs = d['wave_header']['fsValid']
        self.fstop = d['wave_header']['topFullScale']
        self.fsbottom = d['wave_header']['botFullScale']
        if record.wave['version'] in [1,2,3]:
            dims = [d['wave_header']['npnts']] + [0]*(_MAXDIMS-1)
            sfA = [d['wave_header']['hsA']] + [0]*(_MAXDIMS-1)
            sfB = [d['wave_header']['hsB']] + [0]*(_MAXDIMS-1)
            self.data_units = [d['wave_header']['dataUnits']]
            self.axis_units = [d['wave_header']['xUnits']]
        else:
            dims = d['wave_header']['nDim']
            sfA = d['wave_header']['sfA']
            sfB = d['wave_header']['sfB']
            # TODO find example with multiple data units
            self.data_units = [d['data_units'].decode(ENCODING)]
            self.axis_units = [d['dimension_units'].decode(ENCODING)]
        self.data_units.extend(['']*(_MAXDIMS-len(self.data_units)))
        self.data_units = tuple(self.data_units)
        self.axis_units.extend(['']*(_MAXDIMS-len(self.axis_units)))
        self.axis_units = tuple(self.axis_units)
        self.axis = [_numpy.linspace(a,b,c) for a,b,c in zip(sfA, sfB, dims)]
        self.formula = d.get('formula', '')
        self.notes = d.get('note', '')
    def format(self, indent=0):
        if isinstance(self.data, list):
            type,size = "text", "%d"%len(self.data)
        else:
            type,size = "data", "x".join(str(d) for d in self.data.shape)
        return " "*indent+"%s %s (%s)"%(self.name, type, size)

    def __array__(self):
        return self.data

    __repr__ = __str__ = lambda s: "<igor.Wave %s>" % s.format()

class Recreation(IgorObject):
    """
    Contains the experiment's recreation procedures as plain text.
    """
    def __init__(self, data):
        self.data = data
    def format(self, indent=0):
        return " "*indent + "<Recreation>"
class Procedure(IgorObject):
    """
    Contains the experiment's main procedure window text as plain text.
    """
    def __init__(self, data):
        self.data = data
    def format(self, indent=0):
        return " "*indent + "<Procedure>"
class GetHistory(IgorObject):
    """
    Not a real record but rather, a message to go back and read the history text.

    The reason for GetHistory is that IGOR runs Recreation when it loads the
    datafile.  This puts entries in the history that shouldn't be there.  The
    GetHistory entry simply says that the Recreation has run, and the History
    can be restored from the previously saved value.
    """
    def __init__(self, data):
        self.data = data
    def format(self, indent=0):
        return " "*indent + "<GetHistory>"
class PackedFile(IgorObject):
    """
    Contains the data for a procedure file or notebook in packed form.
    """
    def __init__(self, data):
        self.data = data
    def format(self, indent=0):
        return " "*indent + "<PackedFile>"
class Unknown(IgorObject):
    """
    Record type not documented in PTN003/TN003.
    """
    def __init__(self, data, type):
        self.data = data
        self.type = type
    def format(self, indent=0):
        return " "*indent + "<Unknown type %s>"%self.type


class Folder(IgorObject):
    """
    Hierarchical record container.
    """
    def __init__(self, path):
        self.name = path[-1]
        self.path = path
        self.children = []

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.children[key]
        else:
            for r in self.children:
                if isinstance(r, (Folder,Wave)) and r.name == key:
                    return r
            raise KeyError("Folder %s does not exist"%key)

    def __str__(self):
        return "<igor.Folder %s>" % "/".join(self.path)

    __repr__ = __str__

    def append(self, record):
        """
        Add a record to the folder.
        """
        self.children.append(record)
        try:
            # Record may not have a name, the name may be invalid, or it
            # may already be in use.   The noname case will be covered by
            # record.name raising an attribute error.  The others we need
            # to test for explicitly.
            if valid_identifier(record.name) and not hasattr(self, record.name):
                setattr(self, record.name, record)
        except AttributeError:
            pass

    def format(self, indent=0):
        parent = " "*indent+self.name
        children = [r.format(indent=indent+2) for r in self.children]
        return "\n".join([parent]+children)


def loads(s, **kwargs):
    """Load an igor file from string"""
    stream = _io.BytesIO(s)
    return load(stream, **kwargs)

def load(filename, **kwargs):
    """Load an igor file"""
    try:
        packed_experiment = _load(filename)
    except ValueError as e:
        if e.args[0].startswith('not enough data for the next record header'):
            raise IOError('invalid record header; bad pxp file?')
        elif e.args[0].startswith('not enough data for the next record'):
            raise IOError('final record too long; bad pxp file?')
        raise
    return _convert(packed_experiment, **kwargs)

def _convert(packed_experiment, ignore_unknown=True):
    records, filesystem = packed_experiment
    stack = [Folder(path=['root'])]
    for record in records:
        if isinstance(record, _UnknownRecord):
            if ignore_unknown:
                continue
            else:
                r = Unknown(record.data, type=record.header['recordType'])
        elif isinstance(record, _GetHistoryRecord):
            r = GetHistory(record.text)
        elif isinstance(record, _HistoryRecord):
            r = History(record.text)
        elif isinstance(record, _PackedFileRecord):
            r = PackedFile(record.text)
        elif isinstance(record, _ProcedureRecord):
            r = Procedure(record.text)
        elif isinstance(record, _RecreationRecord):
            r = Recreation(record.text)
        elif isinstance(record, _VariablesRecord):
            r = Variables(record)
        elif isinstance(record, _WaveRecord):
            r = Wave(record)
        else:
            r = None

        if isinstance(record, _FolderStartRecord):
            path = stack[-1].path + [
                record.null_terminated_text.decode(ENCODING)]
            folder = Folder(path)
            stack[-1].append(folder)
            stack.append(folder)
        elif isinstance(record, _FolderEndRecord):
            stack.pop()
        elif r is None:
            raise NotImplementedError(record)
        else:
            stack[-1].append(r)
    if len(stack) != 1:
        raise IOError("FolderStart records do not match FolderEnd records")
    return stack[0]
