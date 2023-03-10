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

"Utility functions for handling buffers"

import sys as _sys

import numpy as _numpy


def _ord(byte):
    r"""Convert a byte to an integer.

    >>> buffer = b'\x00\x01\x02'
    >>> [_ord(b) for b in buffer]
    [0, 1, 2]
    """
    if _sys.version_info >= (3,):
        return byte
    else:
        return ord(byte)

def hex_bytes(buffer, spaces=None):
    r"""Pretty-printing for binary buffers.

    >>> hex_bytes(b'\x00\x01\x02\x03\x04')
    '0001020304'
    >>> hex_bytes(b'\x00\x01\x02\x03\x04', spaces=1)
    '00 01 02 03 04'
    >>> hex_bytes(b'\x00\x01\x02\x03\x04', spaces=2)
    '0001 0203 04'
    >>> hex_bytes(b'\x00\x01\x02\x03\x04\x05\x06', spaces=2)
    '0001 0203 0405 06'
    >>> hex_bytes(b'\x00\x01\x02\x03\x04\x05\x06', spaces=3)
    '000102 030405 06'
    """
    hex_bytes = ['{:02x}'.format(_ord(x)) for x in buffer]
    if spaces is None:
        return ''.join(hex_bytes)
    elif spaces is 1:
        return ' '.join(hex_bytes)
    for i in range(len(hex_bytes)//spaces):
        hex_bytes.insert((spaces+1)*(i+1)-1, ' ')
    return ''.join(hex_bytes)

def assert_null(buffer, strict=True):
    r"""Ensure an input buffer is entirely zero.

    >>> import sys
    >>> assert_null(b'')
    >>> assert_null(b'\x00\x00')
    >>> assert_null(b'\x00\x01\x02\x03')
    Traceback (most recent call last):
      ...
    ValueError: 00 01 02 03
    >>> stderr = sys.stderr
    >>> sys.stderr = sys.stdout
    >>> assert_null(b'\x00\x01\x02\x03', strict=False)
    warning: post-data padding not zero: 00 01 02 03
    >>> sys.stderr = stderr
    """
    if buffer and _ord(max(buffer)) != 0:
        hex_string = hex_bytes(buffer, spaces=1)
        if strict:
            raise ValueError(hex_string)
        else:
            _sys.stderr.write(
                'warning: post-data padding not zero: {}\n'.format(hex_string))

# From ReadWave.c
def byte_order(needToReorderBytes):
    little_endian = _sys.byteorder == 'little'
    if needToReorderBytes:
        little_endian = not little_endian
    if little_endian:
        return '<'  # little-endian
    return '>'  # big-endian    

# From ReadWave.c
def need_to_reorder_bytes(version):
    # If the low order byte of the version field of the BinHeader
    # structure is zero then the file is from a platform that uses
    # different byte-ordering and therefore all data will need to be
    # reordered.
    return version & 0xFF == 0

# From ReadWave.c
def checksum(buffer, byte_order, oldcksum, numbytes):
    x = _numpy.ndarray(
        (numbytes/2,), # 2 bytes to a short -- ignore trailing odd byte
        dtype=_numpy.dtype(byte_order+'h'),
        buffer=buffer)
    oldcksum += x.sum()
    if oldcksum > 2**31:  # fake the C implementation's int rollover
        oldcksum %= 2**32
        if oldcksum > 2**31:
            oldcksum -= 2**31
    return oldcksum & 0xffff

def _bytes(obj, encoding='utf-8'):
    """Convert bytes or strings into bytes

    >>> _bytes(b'123')
    '123'
    >>> _bytes('123')
    '123'
    """
    if _sys.version_info >= (3,):
        if isinstance(obj, bytes):
            return obj
        else:
            return bytes(obj, encoding)
    else:
        return bytes(obj)
