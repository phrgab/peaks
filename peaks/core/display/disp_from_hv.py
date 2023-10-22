#Helper function for esimating EF
#Phil King 15/05/2021

import numpy as np
import xarray as xr
from peaks.core.utils.E_shift import Eshift_from_correction
from peaks.core.utils.OOP_method import add_methods

# Function to extract a dispersion from an hv-dep cube of data
@add_methods(xr.DataArray)
def disp_from_hv(data, hv, correct_EF_curve=False):
    '''Return a single dispersion from a hv cube <br>
    Corrects for KE offsets that arise due to the way hv-dep data is stored

    Parameters
    ------------
    data : xr.DataArray
         hv-dep cube of data

    hv : float
        Photon energy to extract

    correct_EF_curve : bool, optional
        Correct for the curvature of the Fermi level <br>
        Based on the supplied normalisation wave in .attrs['EF_correction'] <br>
        Does not convert to BE <br>
        Defaults to False

    Returns
    ------------
    data_out : xr.DataArray
        Single dispersion at photon energy closest to requested one

    Example
    ------------

        '''

    # Check correct type of data being supplied
    if data.attrs['scan_type'] != "hv scan":
        raise Exception("disp_from_hv designed for extracting single dispersion from an hv scan.")

    if data.attrs['eV_type'] == 'kinetic':  # Pull out relevant slice accounting for offset of KE of detector in array

        # Select relevant hv slice and copy the input xarray
        data_out = data.sel(hv=hv, method='nearest').copy(deep=True)

        # Rescale eV axis to get correct KE
        data_out['eV'] = ('eV', data_out.eV.data + data_out.KE_delta.data)

        # Delete KE_delta coordinate as this is no longer meaningful
        del data_out['KE_delta']

        if correct_EF_curve:  # Correct for the slit curvature
            # Extract relevant energy shifts, correction type and raw correction parameters from dataarray
            E_shift, EF_correction_type = Eshift_from_correction(data_out)

            # Convert E_shift into an angle-dependent shift that needs to be applied to straighten the Fermi edge up
            E_shift = max(E_shift) - E_shift

            # Determine relevant eV matrix to interpolate over
            eV_values_tointerp = data_out.eV.data[:, np.newaxis] - E_shift
            eV_xarray = xr.DataArray(eV_values_tointerp, dims=["eV", "theta_par"],
                                     coords={"theta_par": data_out.theta_par.data, "eV": data_out.eV.data})

            # get theta_par values as an xarray
            theta_par_xarray = xr.DataArray(data_out.theta_par.data, dims=["theta_par"],
                                            coords={"theta_par": data_out.theta_par.data})

            # apply the interpolation
            data_out = data_out.interp(eV=eV_xarray, theta_par=theta_par_xarray)

        # Update the hv attribute
        data_out.attrs['hv'] = float(hv)

    else:  # If energy scale is in BE, no KE conversions are required
        raise Exception(
            "disp_from_hv designed for extracting single dispersion from an hv scan before k-space and BE conversion.")

    # Return xarray of the required hv slice
    return data_out


# Function to extract and then plot a dispersion from an hv-dep cube of data
@add_methods(xr.DataArray)
def plot_hv(data, hv, correct_EF_curve=False, **kwargs):
    '''Return a plot of a single dispersion from a hv cube <br>
        Corrects for KE offsets that arise due to the way hv-dep data is stored

        Parameters
        ------------
        data : xr.DataArray
             hv-dep cube of data

        hv : float
            Photon energy to extract

        correct_EF_curve : bool, optional
            Correct for the curvature of the Fermi level <br>
            Based on the supplied normalisation wave in .attrs['EF_correction'] <br>
            Does not convert to BE <br>
            Defaults to False

        **kwargs : optional
            Standard matplotlib calls to pass to plot

        Returns
        ------------
        plot
            Desired plot
    '''

    # Do the slice extraction with disp_from_hv
    data_to_plot = disp_from_hv(data, hv, correct_EF_curve)

    # Make y='eV' the default for the plot
    if 'y' not in kwargs:
        kwargs['y'] = 'eV'

    # Make the plot
    data_to_plot.plot(**kwargs)

    return