#Functions to load data from Bruker D8 diffractometer into an xarray
#Phil King 08/06/2021

import numpy as np
import os
import xarray as xr
from peaks.utils.consts import const

def load_Bruker_data(file, **kwargs):
    '''This function loads x-ray diffraction data from a Bruker D8 Discover

       Input:
           file - Path to the file to be loaded (string)
           **kwargs - Additional options

       Returns:
           data - DataArray with loaded data (xarray)'''

    filename, file_extension = os.path.splitext(file)

    if file_extension == '.xy':
        # Arrays for data
        scan_axis = []
        counts = []

        with open(file) as f:
            lines = f.readlines()
        for i in lines[1:]:
            scan_axis.append(float(i.split(' ')[0]))
            counts.append(float(i.split(' ')[1].split('\n')[0]))

        counts = np.asarray(counts)  # Convert counts into numpy array

        # Check scan type
        scantype = lines[0].split('Scantype: "')[1].split('"')[0]
        # Sometimes has a trailing space - remove this
        try:
            scantype = scantype.split(' ')[0]
        except:
            pass

        if scantype == 'Phi-H':  # Phi scan
            phi = np.asarray(scan_axis)  # Angular axis (np array)

            # create xarray
            data = xr.DataArray(counts, dims=("phi"), coords={"phi": scan_axis})
            data.coords['phi'].attrs = {'units': 'deg'}
        else:
            two_th = np.asarray(scan_axis)  # Angular axis (np array)

            # create xarray
            data = xr.DataArray(counts, dims=("two_th"), coords={"two_th": scan_axis})
            data.coords['two_th'].attrs = {'units': 'deg'}

        data.name = 'Intensity'

        # Check if it is really a reflectivity scan (also a 2Theata-Omega scan in the header)
        if scantype == '2Theta-Omega':
            if data.two_th.max() < 10:  # Quite a small angle, so reflectivity seems a good guess
                scantype = 'Reflectivity'

        # Check if it was really a 2Th-Om scan integrated from the 2D detecotr
        if scantype == 'TwoTheta':
            print(lines[0])
            if 'Integrate frame' in lines[0]:  # This should be a 2Th-Om scan integrated from the detector
                scantype = '2Theta-Omega (from 2D detector)'

        # Add some attributes
        data.attrs['scan_type'] = scantype
        data.attrs['wavelength'] = const.Cu_Ka_lambda  # Cu K-alpha wavelength in Angstroms

        # Time per step
        dwell = float(lines[0].split('TimePerStep: "')[1].split('"')[0])
        data.attrs['dwell'] = dwell
        data.attrs['units'] = 'c.p.s.'

        data = data.chunk()

    return data
