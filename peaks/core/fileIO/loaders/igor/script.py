# Copyright (C) 2012-2016 W. Trevor King <wking@tremily.us>
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

"Common code for scripts distributed with the `igor` package."

from __future__ import absolute_import
import argparse as _argparse
import logging as _logging
import sys as _sys

try:
    import matplotlib as _matplotlib
    import matplotlib.pyplot as _matplotlib_pyplot
except ImportError as _matplotlib_import_error:
    _matplotlib = None

from . import __version__
from . import LOG as _LOG


class Script (object):
    log_levels = [_logging.ERROR, _logging.WARNING, _logging.INFO, _logging.DEBUG]

    def __init__(self, description=None, filetype='IGOR Binary Wave (.ibw) file'):
        self.parser = _argparse.ArgumentParser(description=description)
        self.parser.add_argument(
            '--version', action='version',
            version='%(prog)s {}'.format(__version__))
        self.parser.add_argument(
            '-f', '--infile', metavar='FILE', default='-',
            help='input {}'.format(filetype))
        self.parser.add_argument(
            '-o', '--outfile', metavar='FILE', default='-',
            help='file for ASCII output')
        self.parser.add_argument(
            '-p', '--plot', action='store_const', const=True,
            help='use Matplotlib to plot any IGOR waves')
        self.parser.add_argument(
            '-V', '--verbose', action='count', default=0,
            help='increment verbosity')
        self._num_plots = 0

    def run(self, *args, **kwargs):
        args = self.parser.parse_args(*args, **kwargs)
        if args.infile == '-':
            args.infile = _sys.stdin
        if args.outfile == '-':
            args.outfile = _sys.stdout
        if args.verbose > 1:
            log_level = self.log_levels[min(args.verbose-1, len(self.log_levels)-1)]
            _LOG.setLevel(log_level)
        self._run(args)
        self.display_plots()

    def _run(self, args):
        raise NotImplementedError()

    def plot_wave(self, args, wave, title=None):
        if not args.plot:
            return  # no-op
        if not _matplotlib:
            raise _matplotlib_import_error
        if title is None:
            title = wave['wave']['wave_header']['bname']
        figure = _matplotlib_pyplot.figure()
        axes = figure.add_subplot(1, 1, 1)
        axes.set_title(title)
        try:
            axes.plot(wave['wave']['wData'], 'r.')
        except ValueError as error:
            _LOG.error('error plotting {}: {}'.format(title, error))
            pass
        self._num_plots += 1

    def display_plots(self):
        if self._num_plots:
            _matplotlib_pyplot.show()
