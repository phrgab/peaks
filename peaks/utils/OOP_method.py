#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:07:14 2021

@author: lewishart
"""

from functools import wraps # This convenience func preserves name and docstring

def add_methods(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(*args, **kwargs): 
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator
