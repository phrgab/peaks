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


class Record (object):
    def __init__(self, header, data, byte_order=None):
        self.header = header
        self.data = data
        self.byte_order = byte_order

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, id(self))


class UnknownRecord (Record):
    def __repr__(self):
        return '<{}-{} {}>'.format(
            self.__class__.__name__, self.header['recordType'], id(self))


class UnusedRecord (Record):
    pass


class TextRecord (Record):
    def __init__(self, *args, **kwargs):
        super(TextRecord, self).__init__(*args, **kwargs)
        self.text = bytes(self.data).replace(
            b'\r\n', b'\n').replace(b'\r', b'\n')
        self.null_terminated_text = self.text.split(b'\x00', 1)[0]
