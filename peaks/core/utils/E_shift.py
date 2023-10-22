#Helper function for extracting shifts for EF corrections
#Phil King 15/05/2021


from lmfit.models import PolynomialModel

def Eshift_from_correction(data):
    '''Function determines the relevant energy shifts that will apply for the EF_correction identified in its attributes

    Input:
        data - the relevant dataarray (xarray)

    Output:
        E_shift - the energy shifts
        correction_type - an identifier of the correction type
    '''

    if 'EF_correction' not in data.attrs: #No correction applied
        E_shift = 0
        correction_type = 'None'
    elif data.attrs['EF_correction'] == None: # No correction applied
        E_shift = 0
        correction_type = 'None'
    else:
        try: #This will work if the correction mode is from an xarray, i.e. direct inclusion of EF values along the slit from the reference measurement
            #Theta_par values to interpolate fit array onto
            theta_par_values = data.theta_par.data

            #Interpolate EF values onto data theta_par values
            E_shift_xarray = data.attrs['EF_correction'].interp(theta_par = theta_par_values, kwargs={"fill_value": "extrapolate"})

            #Extract raw E_shift array
            E_shift = E_shift_xarray.data

            #Set correction type
            correction_type = 'array'
        except:
            if type(data.attrs['EF_correction']) == dict: #EF correction from a poly fit
                correction = data.attrs['EF_correction']

                # create a model for the polynomial correction fit vs angle
                correction_fit = PolynomialModel(degree=3)

                # define the correction fit details
                correction_fit_params = correction_fit.make_params(c0=correction['c0'], c1=correction['c1'], c2=correction['c2'], c3=correction['c3'])

                # apply fit along the dispersion theta_par values to produce a list of the Fermi level corrections
                E_shift = correction_fit.eval(params=correction_fit_params, x=data.theta_par.data)

                # Set correction type
                correction_type = 'fit'

            else: # Then this should be some form of float or integer (note, different floats can end up being passed such as np.float64 or float etc., so a little care is needed here
                #Define the shift direct from the float
                E_shift = data.attrs['EF_correction']
                # Set correction type
                correction_type = 'value'

    return E_shift, correction_type