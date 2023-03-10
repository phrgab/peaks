import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
"""
library of user defined fit function:
so far only function returning 1D array

!!! call the independent variable x !!!
you can test your function in test_fit_function.py
"""


def Shirley(y, average = 1, offsetL = 0, offsetR = 0):
    '''
    Shirley background calculated on the entire x axis. no masking.

    Parameters
    ----------
    x : 1Darray-like .
    y : 1Darray-like. same dimension of x
    average : number of point to take in consideration to calulate the average value of the y-start and y-end point .
    offsetL = offsets to subtract to the leftest  y value. useful if the range do not cover completely the left tail of the peak.
    offsetR = offsets to subtract to the rightest  y value. useful if the range do not cover completely the tail of the rightest peak.
    Returns
    -------
    1D-array-like. same dimension of x and y
    '''
    
    start = 0
    end = len(y)
    tol=1e-5
    maxit=10
    average = int(average)
    # Make sure we've been passed arrays and not lists.
    y = np.array(y)
    # Next ensure the energy values are *decreasing* in the array,
    # if not, reverse them.
    if y[0] < y[-1]:
        is_reversed = True
        y = y[::-1]
    else:
        is_reversed = False

    # left and right limits
    yl = np.mean(y[start:start+average])-offsetL
    yr = np.mean(y[end-average:end])-offsetR

    # Max integration index
    imax = end - 1

    # Initial value of the background shape B. The total background S = yr + B,
    # and B is equal to (yl - yr) below start and initially zero above.
    B = np.zeros(y.shape)
    B[:start] = yl - yr
    
    dk = lambda i: sum(0.5 * (y[i:-1] + y[i+1:] - 2*yr - B[i:-1] + B[i+1:]))
    
    it = 0
    while it < maxit:
        # Calculate new k = (yl - yr) / (int_(xl)^(xr) J(x') - yr - B(x') dx')
        ksum = sum(0.5 * (y[:-1] + y[1:] - 2*yr - B[:-1] + B[1:]))
        k = (yl - yr) / ksum
        # Calculate new B
        ysum=np.array(list(map(dk,range(start, end))))
        Bnew = k * ysum
        # If Bnew is close to B, exit.
        if sum(abs(Bnew - B)) < tol:
            B = Bnew
            break
        else:
            B = Bnew
        it += 1
    if it >= maxit:
        print("Shirley: Max iterations exceeded before convergence.")
    if is_reversed:
        return (yr + B)[::-1]
    else:
        return yr + B
    

def convolve_gauss(model, sigma_conv_pxl):
    # gauss convolution to use in lmfit.ModelComposite as binary operator
    return gaussian_filter1d(model, sigma_conv_pxl)


def Sigma2pxl(x, sigma):
    # used in gaussian convolution as dummy function
    # converts sigma_conv in pixel from x axis unit
    return sigma/(abs(x[-1]-x[0])/len(x))


def Fermi_function(x, EF, T):
    """
    This function defines the fermi function used for fitting

    Input:
        x - current energy value
        EF - the Fermi level
        T - the temperature

    Returns:
        fermi_function_value - the calculated value of the Fermi function
    """
    # Boltzmann constant in eV units
    kb = 8.617333262145*10**(-5)
    return expit((EF-x)/(kb*T))
