#Function to correct the spurious slant on the I05 nano-ARPES spectrometer present in 2021/22
#Phil King 05/3/21

import xarray as xr
from peaks.utils.metadata import update_hist
from peaks.utils.OOP_method import add_methods

@add_methods(xr.DataArray)
def slant_correct(data, slant_factor=None):
    ''' This function removes a slant present in the data from the Scienta DA30 (9ES210) on the nano-ARPES
    beamline at Diamond light source in 2021/22.

        Input:
            data - the data to be corrected (xarray)
            slant_factor (optional, default=None) - slant factor correction to use, in degrees/eV.
             Deafult values used if not specified

        Returns:
            data_corrected - the corrected data (dict) '''

    # Make a copy of the data
    data_in = data.copy(deep=True)

    # Determine correction factor:
    if slant_factor is None:
        slant_factor = 8 / data_in.attrs['PE']

    # Energy scaling (unchanged)
    eV_xarray = xr.DataArray(data_in.eV.data, dims=['eV'], coords={'eV': data_in.eV.data})

    # Angle mapping
    theta_par_new = data_in.theta_par - (slant_factor * (data_in.eV-data_in.eV.median()))

    # Do the interpolation
    data_out = data_in.interp({'theta_par': theta_par_new, 'eV': eV_xarray})

    # Update the history
    hist_str = 'Slant correction for Diamond nano-ARPES data applied: ' + str(slant_factor) + ' deg/eV'
    update_hist(data_out, hist_str)

    return data_out