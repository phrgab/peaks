#Helper functions for esimating EF
#Phil King 15/05/2021

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from peaks.core.fit import fit_fermi
from peaks.core.utils.OOP_method import add_methods

#Function to estimate EF from a dispersion (xarray input)
@add_methods(xr.DataArray)
def estimate_EF(disp, fit=True):
    '''Estimate the Fermi level from the data

        Parameters
        ------------
        dispersion : xr.DataArray
             The dispersion (2D dataarray) to extract EDCs from

        ang0 : float
            Angle or k of first EDC to extract

        ang1 : float
            Angle or k of last EDC to extract

        ang_step : float
            Angle/k step between extracted EDCs

        dang : float, optional
            Integration range (total range, integrates +/- dang/2) <br>
            Defaults to single DC

        plot : bool, optional
             Call with 'plot=True' to return a plot

        **kwargs : optional
            Additional arguments to pass to the plotting function <br>
            Offset to define offset between DCs, plus standard matplotlib calls

        Returns
        ------------
        EDCs : xr.DataArray
            xarray DataArray of MDC stack or plot if called with `plot=True`

        '''

    '''Function to 

    Input:
        - disp: dispersion to estimate EF from (xarray)

    Output:
        - Estimated Fermi level (float) '''

    def noise_level(y):
        # calculate the average of nearest neighbour differences to evaluate the noise level of a signal
        NNdist = []
        for i in range(1, len(y)):
            dist = abs(y[i] - y[i - 1])
            NNdist.append(dist)
        noise = np.average(NNdist)
        return noise

    if fit == True:  # Asking for a fit rather than just a crude estimate from the derivative
        EDC = disp.DOS()
        EF = fit_fermi(EDC)['FermiEF']

        EF = np.round(EF,3)

        return EF

    else:  # Estimate it from the first negative peak in the first derivative
        deriv = disp.DOS().smooth(eV=0.01).differentiate('eV')
        y = -deriv.values
        x = deriv.eV.values

        # smooth with gaussian convolution
        smooth_y = gaussian_filter1d(y, 3)

        # calculate the noise level from smoothed y
        noise = noise_level(y)

        # find all the peaks with prominence >= 2.5*noise level and with width at least 3 points
        peaks_index, peaks_details = find_peaks(smooth_y, prominence=noise * 2.5, width=3)

        EF = np.round(x[peaks_index].max(), 3)  # Estimated EF

        return EF

