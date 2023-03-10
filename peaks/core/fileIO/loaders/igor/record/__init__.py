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

"Record parsers for IGOR's packed experiment files."


from .base import Record, UnknownRecord, UnusedRecord
from .variables import VariablesRecord
from .history import HistoryRecord, RecreationRecord, GetHistoryRecord
from .wave import WaveRecord
from .procedure import ProcedureRecord
from .packedfile import PackedFileRecord
from .folder import FolderStartRecord, FolderEndRecord


# From PackedFile.h
RECORD_TYPE = {
    0: UnusedRecord,
    1: VariablesRecord,
    2: HistoryRecord,
    3: WaveRecord,
    4: RecreationRecord,
    5: ProcedureRecord,
    6: UnusedRecord,
    7: GetHistoryRecord,
    8: PackedFileRecord,
    9: FolderStartRecord,
    10: FolderEndRecord,
    }
