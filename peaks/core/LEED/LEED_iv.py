# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:51:12 2021

@author: lsh8
"""

import xarray as xr
import numpy as np
from peaks.core.utils.OOP_method import add_methods

@add_methods(xr.DataArray)
def LEED_iv(data, Bragg = None):
    #data = self
    
    if isinstance(data, xr.core.dataarray.DataArray):
        #Create single iv graph
        if Bragg is not None:
            pass
        elif hasattr['data', Bragg]:
            Bragg = data.attrs['Bragg']
        else:
            raise Exception("No Bragg peaks")
        
        if np.ndim(Bragg) == 1:
            iv = data.sel(q_x = Bragg[0], method = 'nearest').sel(q_y = Bragg[1], method = 'nearest')
        elif np.ndmin(Bragg) == 2:
            adict = {}
            for i in range(len(Bragg[:,0])):
                iv_darray = data.sel(q_x = Bragg[i][0], method = 'nearest').sel(q_y = Bragg[i][1], method = 'nearest')
                adict[i]=iv_darray
                
                iv = xr.Dataset(data_vars=adict)
                
    return iv
                
