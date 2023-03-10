# Copyright (C) 2012 W. Trevor King <wking@tremily.us>
#
# This file is part of igor.
#
# igor is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# igor is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with igor.  If not, see <http://www.gnu.org/licenses/>.

"""Structure and Field classes for declaring structures

There are a few formats that can be used to represent the same data, a
binary packed format with all the data in a buffer, a linearized
format with each field in a single Python list, and a nested format
with each field in a hierarchy of Python dictionaries.
"""

from __future__ import absolute_import
import io as _io
import logging as _logging
import pprint as _pprint
import struct as _struct

import numpy as _numpy

from . import LOG as _LOG


class Field (object):
    """Represent a Structure field.

    The format argument can be a format character from the ``struct``
    documentation (e.g., ``c`` for ``char``, ``h`` for ``short``, ...)
    or ``Structure`` instance (for building nested structures).

    Examples
    --------

    >>> from pprint import pprint
    >>> import numpy

    Example of an unsigned short integer field:

    >>> time = Field(
    ...     'I', 'time', default=0, help='POSIX time')
    >>> time.arg_count
    1
    >>> list(time.pack_data(1))
    [1]
    >>> list(time.pack_item(2))
    [2]
    >>> time.unpack_data([3])
    3
    >>> time.unpack_item([4])
    4

    Example of a multi-dimensional float field:

    >>> data = Field(
    ...     'f', 'data', help='example data', count=(2,3,4), array=True)
    >>> data.arg_count
    24
    >>> list(data.indexes())  # doctest: +ELLIPSIS
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 1, 0], ..., [1, 2, 3]]
    >>> list(data.pack_data(
    ...     [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    ...      [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ...     )  # doctest: +ELLIPSIS
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ..., 19, 20, 21, 22, 23]
    >>> list(data.pack_item(3))
    [3]
    >>> data.unpack_data(range(data.arg_count))
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> data.unpack_item([3])
    3

    Example of a nested structure field:

    >>> run = Structure('run', fields=[time, data])
    >>> runs = Field(run, 'runs', help='pair of runs', count=2, array=True)
    >>> runs.arg_count  # = 2 * (1 + 24)
    50
    >>> data1 = numpy.arange(data.arg_count).reshape(data.count)
    >>> data2 = data1 + data.arg_count
    >>> list(runs.pack_data(
    ...     [{'time': 100, 'data': data1},
    ...      {'time': 101, 'data': data2}])
    ...     )  # doctest: +ELLIPSIS
    [100, 0, 1, 2, ..., 22, 23, 101, 24, 25, ..., 46, 47]
    >>> list(runs.pack_item({'time': 100, 'data': data1})
    ...     )  # doctest: +ELLIPSIS
    [100, 0, 1, 2, ..., 22, 23]
    >>> pprint(runs.unpack_data(range(runs.arg_count)))
    [{'data': array([[[ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12]],
    <BLANKLINE>
           [[13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24]]]),
      'time': 0},
     {'data': array([[[26, 27, 28, 29],
            [30, 31, 32, 33],
            [34, 35, 36, 37]],
    <BLANKLINE>
           [[38, 39, 40, 41],
            [42, 43, 44, 45],
            [46, 47, 48, 49]]]),
      'time': 25}]
    >>> pprint(runs.unpack_item(range(runs.structure_count)))
    {'data': array([[[ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12]],
    <BLANKLINE>
           [[13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24]]]),
     'time': 0}

    If you don't give enough values for an array field, the remaining
    values are filled in with their defaults.

    >>> list(data.pack_data(
    ...     [[[0, 1, 2, 3], [4, 5, 6]], [[10]]]))  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: no default for <Field data ...>
    >>> data.default = 0
    >>> list(data.pack_data(
    ...     [[[0, 1, 2, 3], [4, 5, 6]], [[10]]]))
    [0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    See Also
    --------
    Structure
    """
    def __init__(self, format, name, default=None, help=None, count=1,
                 array=False):
        self.format = format
        self.name = name
        self.default = default
        self.help = help
        self.count = count
        self.array = array
        self.setup()

    def setup(self):
        """Setup any dynamic properties of a field.

        Use this method to recalculate dynamic properities after
        changing the basic properties set during initialization.
        """
        _LOG.debug('setup {}'.format(self))
        self.item_count = _numpy.prod(self.count)  # number of item repeats
        if not self.array and self.item_count != 1:
            raise ValueError(
                '{} must be an array field to have a count of {}'.format(
                    self, self.count))
        if isinstance(self.format, Structure):
            self.structure_count = sum(
                f.arg_count for f in self.format.fields)
            self.arg_count = self.item_count * self.structure_count
        elif self.format == 'x':
            self.arg_count = 0  # no data in padding bytes
        else:
            self.arg_count = self.item_count  # struct.Struct format args

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<{} {} {}>'.format(
            self.__class__.__name__, self.name, id(self))

    def indexes(self):
        """Iterate through indexes to a possibly multi-dimensional array"""
        assert self.array, self
        try:
            i = [0] * len(self.count)
        except TypeError:  # non-iterable count
            for i in range(self.count):
                yield i
        else:
            for i in range(self.item_count):
                index = []
                for j,c in enumerate(reversed(self.count)):
                    index.insert(0, i % c)
                    i //= c
                yield index

    def pack_data(self, data=None):
        """Linearize a single field's data to a flat list.

        If the field is repeated (count > 1), the incoming data should
        be iterable with each iteration returning a single item.
        """
        if self.array:
            if data is None:
                data = []
            if hasattr(data, 'flat'):  # take advantage of numpy's ndarray.flat
                items = 0
                for item in data.flat:
                    items += 1
                    for arg in self.pack_item(item):
                        yield arg
                if items < self.item_count:
                    if f.default is None:
                        raise ValueError(
                            'no default for {}.{}'.format(self, f))
                    for i in range(self.item_count - items):
                        yield f.default
            else:
                for index in self.indexes():
                    try:
                        if isinstance(index, int):
                            item = data[index]
                        else:
                            item = data
                            for i in index:
                                item = item[i]
                    except IndexError:
                        item = None
                    for arg in self.pack_item(item):
                        yield arg
        else:
            for arg in self.pack_item(data):
                yield arg

    def pack_item(self, item=None):
        """Linearize a single count of the field's data to a flat iterable
        """
        if isinstance(self.format, Structure):
            for i in self.format._pack_item(item):
                yield i
        elif item is None:
            if self.default is None:
                raise ValueError('no default for {}'.format(self))
            yield self.default
        else:
            yield item

    def unpack_data(self, data):
        """Inverse of .pack_data"""
        _LOG.debug('unpack {} for {} {}'.format(data, self, self.format))
        iterator = iter(data)
        try:
            items = [next(iterator) for i in range(self.arg_count)]
        except StopIteration:
            raise ValueError('not enough data to unpack {}'.format(self))
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError('too much data to unpack {}'.format(self))
        if isinstance(self.format, Structure):
            # break into per-structure clumps
            s = self.structure_count
            items = zip(*[items[i::s] for i in range(s)])
        else:
            items = [[i] for i in items]
        unpacked = [self.unpack_item(i) for i in items]
        if self.arg_count:
            count = self.count
        else:
            count = 0  # padding bytes, etc.
        if not self.array:
            assert count == 1, (self, self.count)
            return unpacked[0]
        if isinstance(self.format, Structure):
            try:
                len(self.count)
            except TypeError:
                pass
            else:
                raise NotImplementedError('reshape Structure field')
        else:
            unpacked = _numpy.array(unpacked)
            _LOG.debug('reshape {} data from {} to {}'.format(
                    self, unpacked.shape, count))
            unpacked = unpacked.reshape(count)
        return unpacked

    def unpack_item(self, item):
        """Inverse of .unpack_item"""
        if isinstance(self.format, Structure):
            return self.format._unpack_item(item)
        else:
            assert len(item) == 1, item
            return item[0]


class DynamicField (Field):
    """Represent a DynamicStructure field with a dynamic definition.

    Adds the methods ``.pre_pack``, ``pre_unpack``, and
    ``post_unpack``, all of which are called when a ``DynamicField``
    is used by a ``DynamicStructure``.  Each method takes the
    arguments ``(parents, data)``, where ``parents`` is a list of
    ``DynamicStructure``\s that own the field and ``data`` is a dict
    hierarchy of the structure data.

    See the ``DynamicStructure`` docstring for the exact timing of the
    method calls.

    See Also
    --------
    Field, DynamicStructure
    """
    def pre_pack(self, parents, data):
        "Prepare to pack."
        pass

    def pre_unpack(self, parents, data):
        "React to previously unpacked data"
        pass

    def post_unpack(self, parents, data):
        "React to our own data"
        pass

    def _get_structure_data(self, parents, data, structure):
        """Extract the data belonging to a particular ancestor structure.
        """
        d = data
        s = parents[0]
        if s == structure:
            return d
        for p in parents[1:]:
            for f in s.fields:
                if f.format == p:
                    s = p
                    d = d[f.name]
                    break
            assert s == p, (s, p)
            if p == structure:
                break
        return d


class Structure (_struct.Struct):
    r"""Represent a C structure.

    A convenient wrapper around struct.Struct that uses Fields and
    adds dict-handling methods for transparent name assignment.

    See Also
    --------
    Field

    Examples
    --------

    >>> import array
    >>> from pprint import pprint

    Represent the C structures::

        struct run {
          unsigned int time;
          short data[2][3];
        };

        struct experiment {
          unsigned short version;
          struct run runs[2];
        };

    As:

    >>> time = Field('I', 'time', default=0, help='POSIX time')
    >>> data = Field(
    ...     'h', 'data', default=0, help='example data', count=(2,3),
    ...     array=True)
    >>> run = Structure('run', fields=[time, data])
    >>> version = Field(
    ...     'H', 'version', default=1, help='example version')
    >>> runs = Field(run, 'runs', help='pair of runs', count=2, array=True)
    >>> experiment = Structure('experiment', fields=[version, runs])

    The structures automatically calculate the flattened data format:

    >>> run.format
    '@Ihhhhhh'
    >>> run.size  # 4 + 2*3*2
    16
    >>> experiment.format
    '@HIhhhhhhIhhhhhh'
    >>> experiment.size  # 2 + 2 + 2*(4 + 2*3*2)
    36

    The first two elements in the above size calculation are 2 (for
    the unsigned short, 'H') and 2 (padding so the unsigned int aligns
    with a 4-byte block).  If you select a byte ordering that doesn't
    mess with alignment and recalculate the format, the padding goes
    away and you get:

    >>> experiment.set_byte_order('>')
    >>> experiment.get_format()
    '>HIhhhhhhIhhhhhh'
    >>> experiment.size
    34

    You can read data out of any object supporting the buffer
    interface:

    >>> b = array.array('B', range(experiment.size))
    >>> d = experiment.unpack_from(buffer=b)
    >>> pprint(d)
    {'runs': [{'data': array([[1543, 2057, 2571],
           [3085, 3599, 4113]]),
               'time': 33752069},
              {'data': array([[5655, 6169, 6683],
           [7197, 7711, 8225]]),
               'time': 303240213}],
     'version': 1}
    >>> [hex(x) for x in d['runs'][0]['data'].flat]
    ['0x607L', '0x809L', '0xa0bL', '0xc0dL', '0xe0fL', '0x1011L']

    You can also read out from strings:

    >>> d = experiment.unpack(b.tostring())
    >>> pprint(d)
    {'runs': [{'data': array([[1543, 2057, 2571],
           [3085, 3599, 4113]]),
               'time': 33752069},
              {'data': array([[5655, 6169, 6683],
           [7197, 7711, 8225]]),
               'time': 303240213}],
     'version': 1}

    If you don't give enough values for an array field, the remaining
    values are filled in with their defaults.

    >>> experiment.pack_into(buffer=b, data=d)
    >>> b.tostring()[:17]
    '\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10'
    >>> b.tostring()[17:]
    '\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f !'
    >>> run0 = d['runs'].pop(0)
    >>> b = experiment.pack(data=d)
    >>> b[:17]
    '\x00\x01\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f '
    >>> b[17:]
    '!\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'

    If you set ``count=0``, the field is ignored.

    >>> experiment2 = Structure('experiment', fields=[
    ...     version, Field('f', 'ignored', count=0, array=True), runs],
    ...     byte_order='>')
    >>> experiment2.format
    '>HIhhhhhhIhhhhhh'
    >>> d = experiment2.unpack(b)
    >>> pprint(d)
    {'ignored': array([], dtype=float64),
     'runs': [{'data': array([[5655, 6169, 6683],
           [7197, 7711, 8225]]),
               'time': 303240213},
              {'data': array([[0, 0, 0],
           [0, 0, 0]]), 'time': 0}],
     'version': 1}
    >>> del d['ignored']
    >>> b2 = experiment2.pack(d)
    >>> b2 == b
    True
    """
    _byte_order_symbols = '@=<>!'

    def __init__(self, name, fields, byte_order='@'):
        # '=' for native byte order, standard size and alignment
        # See http://docs.python.org/library/struct for details
        self.name = name
        self.fields = fields
        self.byte_order = byte_order
        self.setup()

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<{} {} {}>'.format(
            self.__class__.__name__, self.name, id(self))

    def setup(self):
        """Setup any dynamic properties of a structure.

        Use this method to recalculate dynamic properities after
        changing the basic properties set during initialization.
        """
        _LOG.debug('setup {!r}'.format(self))
        self.set_byte_order(self.byte_order)
        self.get_format()

    def set_byte_order(self, byte_order):
        """Allow changing the format byte_order on the fly.
        """
        _LOG.debug('set byte order for {!r} to {}'.format(self, byte_order))
        self.byte_order = byte_order
        for field in self.fields:
            if isinstance(field.format, Structure):
                field.format.set_byte_order(byte_order)

    def get_format(self):
        format = self.byte_order + ''.join(self.sub_format())
        # P format only allowed for native byte ordering
        # Convert P to I for ILP32 compatibility when running on a LP64.
        format = format.replace('P', 'I')
        try:
            super(Structure, self).__init__(format=format)
        except _struct.error as e:
            raise ValueError((e, format))
        return format

    def sub_format(self):
        _LOG.debug('calculate sub-format for {!r}'.format(self))
        for field in self.fields:
            if isinstance(field.format, Structure):
                field_format = list(
                    field.format.sub_format()) * field.item_count
            else:
                field_format = [field.format]*field.item_count
            for fmt in field_format:
                yield fmt

    def _pack_item(self, item=None):
        """Linearize a single count of the structure's data to a flat iterable
        """
        if item is None:
            item = {}
        for f in self.fields:
            try:
                data = item[f.name]
            except TypeError:
                raise ValueError((f.name, item))
            except KeyError:
                data = None
            for arg in f.pack_data(data):
                yield arg

    def _unpack_item(self, args):
        """Inverse of ._unpack_item"""
        data = {}
        iterator = iter(args)
        for f in self.fields:
            try:
                items = [next(iterator) for i in range(f.arg_count)]
            except StopIteration:
                raise ValueError('not enough data to unpack {}.{}'.format(
                        self, f))
            data[f.name] = f.unpack_data(items)
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise ValueError('too much data to unpack {}'.format(self))
        return data

    def pack(self, data):
        args = list(self._pack_item(data))
        try:
            return super(Structure, self).pack(*args)
        except:
            raise ValueError(self.format)

    def pack_into(self, buffer, offset=0, data={}):
        args = list(self._pack_item(data))
        return super(Structure, self).pack_into(
            buffer, offset, *args)

    def unpack(self, *args, **kwargs):
        args = super(Structure, self).unpack(*args, **kwargs)
        return self._unpack_item(args)

    def unpack_from(self, buffer, offset=0, *args, **kwargs):
        _LOG.debug(
            'unpack {!r} for {!r} ({}, offset={}) with {} ({})'.format(
                buffer, self, len(buffer), offset, self.format, self.size))
        args = super(Structure, self).unpack_from(
            buffer, offset, *args, **kwargs)
        return self._unpack_item(args)

    def get_field(self, name):
        return [f for f in self.fields if f.name == name][0]


class DebuggingStream (object):
    def __init__(self, stream):
        self.stream = stream

    def read(self, size):
        data = self.stream.read(size)
        _LOG.debug('read {} from {}: ({}) {!r}'.format(
                size, self.stream, len(data), data))
        return data


class DynamicStructure (Structure):
    r"""Represent a C structure field with a dynamic definition.

    Any dynamic fields have their ``.pre_pack`` called before any
    structure packing is done.  ``.pre_unpack`` is called for a
    particular field just before that field's ``.unpack_data`` call.
    ``.post_unpack`` is called for a particular field just after
    ``.unpack_data``.  If ``.post_unpack`` returns ``True``, the same
    field is unpacked again.

    Examples
    --------

    >>> from pprint import pprint

    This allows you to define structures where some portion of the
    global structure depends on earlier data.  For example, in the
    quasi-C structure::

        struct vector {
          unsigned int length;
          short data[length];
        };

    You can generate a Python version of this structure in two ways,
    with a dynamic ``length``, or with a dynamic ``data``.  In both
    cases, the required methods are the same, the only difference is
    where you attach them.

    >>> def packer(self, parents, data):
    ...     vector_structure = parents[-1]
    ...     vector_data = self._get_structure_data(
    ...         parents, data, vector_structure)
    ...     length = len(vector_data['data'])
    ...     vector_data['length'] = length
    ...     data_field = vector_structure.get_field('data')
    ...     data_field.count = length
    ...     data_field.setup()
    >>> def unpacker(self, parents, data):
    ...     vector_structure = parents[-1]
    ...     vector_data = self._get_structure_data(
    ...         parents, data, vector_structure)
    ...     length = vector_data['length']
    ...     data_field = vector_structure.get_field('data')
    ...     data_field.count = length
    ...     data_field.setup()

    >>> class DynamicLengthField (DynamicField):
    ...     def pre_pack(self, parents, data):
    ...         packer(self, parents, data)
    ...     def post_unpack(self, parents, data):
    ...         unpacker(self, parents, data)
    >>> dynamic_length_vector = DynamicStructure('vector',
    ...     fields=[
    ...         DynamicLengthField('I', 'length'),
    ...         Field('h', 'data', count=0, array=True),
    ...         ],
    ...     byte_order='>')
    >>> class DynamicDataField (DynamicField):
    ...     def pre_pack(self, parents, data):
    ...         packer(self, parents, data)
    ...     def pre_unpack(self, parents, data):
    ...         unpacker(self, parents, data)
    >>> dynamic_data_vector = DynamicStructure('vector',
    ...     fields=[
    ...         Field('I', 'length'),
    ...         DynamicDataField('h', 'data', count=0, array=True),
    ...         ],
    ...     byte_order='>')

    >>> b = b'\x00\x00\x00\x02\x01\x02\x03\x04'
    >>> d = dynamic_length_vector.unpack(b)
    >>> pprint(d)
    {'data': array([258, 772]), 'length': 2}
    >>> d = dynamic_data_vector.unpack(b)
    >>> pprint(d)
    {'data': array([258, 772]), 'length': 2}

    >>> d['data'] = [1,2,3,4]
    >>> dynamic_length_vector.pack(d)
    '\x00\x00\x00\x04\x00\x01\x00\x02\x00\x03\x00\x04'
    >>> dynamic_data_vector.pack(d)
    '\x00\x00\x00\x04\x00\x01\x00\x02\x00\x03\x00\x04'

    The implementation is a good deal more complicated than the one
    for ``Structure``, because we must make multiple calls to
    ``struct.Struct.unpack`` to unpack the data.
    """
    #def __init__(self, *args, **kwargs):
    #     pass #self.parent = ..

    def _pre_pack(self, parents=None, data=None):
        if parents is None:
            parents = [self]
        else:
            parents = parents + [self]
        for f in self.fields:
            if hasattr(f, 'pre_pack'):
                _LOG.debug('pre-pack {}'.format(f))
                f.pre_pack(parents=parents, data=data)
            if isinstance(f.format, DynamicStructure):
                _LOG.debug('pre-pack {!r}'.format(f.format))
                f._pre_pack(parents=parents, data=data)

    def pack(self, data):
        self._pre_pack(data=data)
        self.setup()
        return super(DynamicStructure, self).pack(data)

    def pack_into(self, buffer, offset=0, data={}):
        self._pre_pack(data=data)
        self.setup()
        return super(DynamicStructure, self).pack_into(
            buffer=buffer, offset=offset, data=data)

    def unpack_stream(self, stream, parents=None, data=None, d=None):
        # `d` is the working data directory
        if data is None:
            parents = [self]
            data = d = {}
            if _LOG.level <= _logging.DEBUG:
                stream = DebuggingStream(stream)
        else:
            parents = parents + [self]

        for f in self.fields:
            _LOG.debug('parsing {!r}.{} (count={}, item_count={})'.format(
                    self, f, f.count, f.item_count))
            if _LOG.level <= _logging.DEBUG:
                _LOG.debug('data:\n{}'.format(_pprint.pformat(data)))
            if hasattr(f, 'pre_unpack'):
                _LOG.debug('pre-unpack {}'.format(f))
                f.pre_unpack(parents=parents, data=data)

            if hasattr(f, 'unpack'):  # override default unpacking
                _LOG.debug('override unpack for {}'.format(f))
                d[f.name] = f.unpack(stream)
                continue

            # setup for unpacking loop
            if isinstance(f.format, Structure):
                f.format.set_byte_order(self.byte_order)
                f.setup()
                f.format.setup()
                if isinstance(f.format, DynamicStructure):
                    if f.array:
                        d[f.name] = []
                        for i in range(f.item_count):
                            x = {}
                            d[f.name].append(x)
                            f.format.unpack_stream(
                                stream, parents=parents, data=data, d=x)
                    else:
                        assert f.item_count == 1, (f, f.count)
                        d[f.name] = {}
                        f.format.unpack_stream(
                            stream, parents=parents, data=data, d=d[f.name])
                    if hasattr(f, 'post_unpack'):
                        _LOG.debug('post-unpack {}'.format(f))
                        repeat = f.post_unpack(parents=parents, data=data)
                        if repeat:
                            raise NotImplementedError(
                                'cannot repeat unpack for dynamic structures')
                    continue
            if isinstance(f.format, Structure):
                _LOG.debug('parsing {} bytes for {}'.format(
                        f.format.size, f.format.format))
                bs = [stream.read(f.format.size) for i in range(f.item_count)]
                def unpack():
                    f.format.set_byte_order(self.byte_order)
                    f.setup()
                    f.format.setup()
                    x = [f.format.unpack_from(b) for b in bs]
                    if not f.array:
                        assert len(x) == 1, (f, f.count, x)
                        x = x[0]
                    return x
            else:
                field_format = self.byte_order + f.format*f.item_count
                field_format = field_format.replace('P', 'I')
                try:
                    size = _struct.calcsize(field_format)
                except _struct.error as e:
                    _LOG.error(e)
                    _LOG.error('{}.{}: {}'.format(self, f, field_format))
                    raise
                _LOG.debug('parsing {} bytes for preliminary {}'.format(
                        size, field_format))
                raw = stream.read(size)
                if len(raw) < size:
                    raise ValueError(
                        'not enough data to unpack {}.{} ({} < {})'.format(
                            self, f, len(raw), size))
                def unpack():
                    field_format = self.byte_order + f.format*f.item_count
                    field_format = field_format.replace('P', 'I')
                    _LOG.debug('parse previous bytes using {}'.format(
                            field_format))
                    struct = _struct.Struct(field_format)
                    items = struct.unpack(raw)
                    return f.unpack_data(items)

            # unpacking loop
            repeat = True
            while repeat:
                d[f.name] = unpack()
                if hasattr(f, 'post_unpack'):
                    _LOG.debug('post-unpack {}'.format(f))
                    repeat = f.post_unpack(parents=parents, data=data)
                else:
                    repeat = False
                if repeat:
                    _LOG.debug('repeat unpack for {}'.format(f))

        return data

    def unpack(self, string):
        stream = _io.BytesIO(string)
        return self.unpack_stream(stream)

    def unpack_from(self, buffer, offset=0, *args, **kwargs):
        args = super(Structure, self).unpack_from(
            buffer, offset, *args, **kwargs)
        return self._unpack_item(args)
