# peaks/_lazy_import.py
"""_lzay_import.py

This module provides a class that allows for lazy importing of modules, i.e. to only import a module when it is called.

Classes:
- LazyImport: Class that allows for lazy importing of modules.

"""

import importlib

__all__ = ["LazyImport"]


class LazyImport:
    def __init__(self, name):
        self._name = name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._name)

    def __getattr__(self, attr):
        self._load()
        return getattr(self._module, attr)
