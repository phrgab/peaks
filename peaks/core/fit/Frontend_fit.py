#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 09:29:21 2021

frontend class and methods used for fitting and fit1d GUI
based on lmfit.Model and iwidgets for simple UI
@author: ta50
"""

import ipywidgets as widgets
from ipywidgets import interactive_output, fixed,Layout
from lmfit.models import *
import lmfit.printfuncs
#file where to store uesr difine functions
from peaks.core.fit.User_Fit_function import  *
import matplotlib.pyplot as plt
import xarray as xr

# list here all the built-in function names in lmfit modlue. these are going to be displayed in the Qtwidget (self.widg_mod)
Special_list = ['None','Linear','Quadratic','Shirley','Gauss_convolution','Fermi_function']

Peak_list = ['None','Lorentzian','Gaussian','Voigt','Doniach']

def fit_function_list():
    #just return the Default_model_list available
    return [Peak_list,Special_list]


class FitFunc():
    """
    class defining a Fit function that can be a single componment or a complex model composed by multiple fit functions
    """
    def __init__(self,
                 func_type = 'None',
                 prefix = '',
                 color = '#70a9e7'):

        # funtion type in Default_model_list, default = None or element in
        self.func_type = func_type
        # prefix that goes in front of all the parameters names. keep it short
        self.prefix = prefix
        # color used for plotting
        self.color = color
        #constrain expression
        self.constr = {}
        # components that compose the complex model
        self.components={}
        # lmfit model object
        self.model = None
        # lmfit parameters object
        self.params = {}
        #last evaluation
        self.eval = []
        self.create_model()

    def _repr_html_(self):
        """Return a HTML representation of parameters data."""
        html = []

        html.append(str('<h2>'+str(self.model.name)+'</h2>'))
        html.append(lmfit.printfuncs.params_html_table(self.params))
        return ''.join(html)




    def Special_list(self):
        #just return the Default_model_list available
        return(Special_list)

    def create_model(self):
        """
        initialise the lmfit model and lmfit parameters for a single component
        """
        if self.func_type =='Linear':
            self.model = LinearModel(prefix = self.prefix)
            self.model.set_param_hint('slope', value=0)
        if self.func_type =='Quadratic':
            self.model = QuadraticModel(prefix = self.prefix)
        if self.func_type =='Shirley':
            self.model = Model(Shirley,independent_vars = ['y'], prefix = self.prefix)
            self.model.set_param_hint('average', value=3.0)
        if self.func_type =='Lorentzian':
            self.model = LorentzianModel(prefix = self.prefix)
            self.model.set_param_hint('amplitude', min = 0)
        if self.func_type =='Gaussian':
            self.model = GaussianModel(prefix = self.prefix)
            self.model.set_param_hint('amplitude', min = 0)
        if self.func_type =='Voigt':
            self.model = VoigtModel(prefix = self.prefix)
            self.model.set_param_hint('amplitude', min = 0)
        if self.func_type =='Doniach':
            self.model = DoniachModel(prefix = self.prefix)
            self.model.set_param_hint('amplitude', min = 0)
        if self.func_type =='Gauss_convolution':
            self.model = Model(Sigma2pxl, prefix = self.prefix)
            self.model.set_param_hint('Convsigma', value=0.001, vary=False)
        if self.func_type == 'Fermi_function':
            self.model = Model(Fermi_function, prefix=self.prefix)
            self.model.set_param_hint('T', value=100, vary=False)
            self.model.set_param_hint('EF', value=0, vary=False)

        if self.func_type !='None':
            self.create_params()
        else:
            self.model = None


    def create_params(self):
        """
        create the default parameters for the composite model
        create fake constraints
        """
        if self.func_type =='None':
            print (self.prefix,'model name  = None','impossible to create parameters on None model')
        else:
            self.cons = {}
            self.params = self.model.make_params()
            for key in self.params.keys():
                self.cons[key] = self.params[key].expr

    def _update_ipwidget(self,param_key,value,hold,min,max):
        """
        private utility for widget_param()
        """
        self.params[param_key].set(value=value,vary = not hold, max = max, min = min)

    def widget_param(self):
        """
        create widgets as UI for parameters initialisation.
        param_key = parameter_key
        value -->  widgets.FloatText
        hold --> widgets.Checkbox
        constrain --> widgets.Textarea
        """
        try:
            self.params
        except:
            print("parameter not found, be sure you have already create the parameters with create_params() or lmfit.parameters.make_params()")

        for param_key in self.params.keys():
            if not 'fwhm' in param_key:
                if not 'height' in param_key:
                    layout_widg=Layout(width='20%')
                    layout_hold=Layout(width='20%')
                    layout_box = widgets.Layout(
                     flex_flow='row',
                     align_items ='flex-start')
                    par_val=widgets.FloatText(
                        value=self.params[param_key].value,
                        description=param_key,
                        layout = layout_widg,
                        disabled= False)

                    par_hold=widgets.Checkbox(
                        value= not self.params[param_key].vary,
                        description='hold',
                        layout = layout_hold,
                        disabled=False)

                    par_min=widgets.FloatText(
                        value=self.params[param_key].min,
                        description='min:',
                        disabled=False,
                        layout=layout_widg)

                    par_max=widgets.FloatText(
                        value=self.params[param_key].max,
                        description='max:',
                        disabled=False,
                        layout=layout_widg)

                    ui = widgets.Box([par_val, par_hold,par_min,par_max], layout = layout_box)

                    out1 = interactive_output(self._update_ipwidget,{'param_key':fixed(param_key),
                                                                     'value': par_val, 'hold': par_hold, 'min': par_min, 'max': par_max})
                    display(ui,out1)


    def evaluate(self, x=[], y=[]):
        """
        evaluate the fit function on x y
        """
        #needs sanity check on model and params
        try:
            x=np.array(x)
            y=np.array(y)
        except:
            raise

        try:
            self.model
            self.params
        except:
            raise

        self.eval = self.model.eval(x=x, y=y, params=self.params)

    def fit(self, x=[], y=[]):
        """
        run fit function on x y
        """
        #needs sanity check on model and params
        try:
            x=np.array(x)
            y=np.array(y)
        except:
            raise
        return self.model.fit(x=x, data=y, params=self.params)


    def plot_components(self, x=[], y = []):
        """
        plot model tot with its components as last evaluated
        x = inedpendent variable
        """
        plt.scatter(x,y,c='k',s =1)
        for key in self.components.keys():
            if key not in ['Conv']:
                self.components[key].evaluate(x=x,y=y)
                plt.plot(x, self.components[key].eval, label=self.components[key].prefix, c=self.components[key].color)
            if key in ['Conv']:
                print(self.components[key].params['Convsigma'])

        self.evaluate(x=x,y=y)
        plt.plot(x, self.eval, c='r', label='model_tot')
        plt.xlabel(x.name)
        plt.ylabel(y.name)
        plt.legend()





def create_numeric_bkg(y, bkg):
    """
    Create a numeric background used in case of (Shirley, ...).
    this is a fake model in the sense it doesn't have any parameter to fit.
    the background is evaluated and passed to Numeric_bkg as array that remains hidden to lmfit (this is why is wrapped and bkg_data)

    :param y: 1D array
    :param bkg:
    :return: numeric_bkg lmfit.model that has x as independent variable and no parameters
    """
    if bkg.func_type == 'Shirley':
        # evaluate the Shirley bkg
        bkg_data = bkg.model.eval(y=y, params=bkg.params)

        # create a fake function for the Model definition with x independent variable and no other input args = parameters in lmfit model)
        def Numeric_bkg(x):
            return bkg_data

        # redefine bkg with lmfit model definition
        final_bkg = FitFunc(bkg.func_type, bkg.prefix,bkg.color)
        # lmfit model needs at least one independent_vars
        final_bkg.model = Model(Numeric_bkg, prefix = bkg.prefix, independent_vars=['x'])
        return final_bkg

def compose_model(y=[], components = [FitFunc()], background = FitFunc(), Fermi = FitFunc(), Convolution = FitFunc()):
    """
    compose a total model as {(sum of the components+ background)*Fermi_function}Convolution
    if a component is None it will not included in the composit model
    Args:
        components: list of Fitfunc for components
        background: Fitfunc for background
        Fermi: Fitfunc for Fermi
        Convolution: Fitfunc for Convolution

    Returns: composed model as Fitfunc instance

    """

    # sanity check on inputs
    for c in components:
        if not type(c) in [type(FitFunc()), type(None)]:
            print('comp',c)
            raise TypeError('a element in the list of components is not a Fitfunc instance')

    if not type(background) in [type(FitFunc()), type(None)]:
        raise TypeError('background is not a Fitfunc instance')

    if not type(Fermi) in [type(FitFunc()), type(None)]:
        raise TypeError('Fermi is not a Fitfunc instance')

    if not type(Convolution) in [type(FitFunc()), type(None)]:
        raise TypeError('Convolution is not a Fitfunc instance')


    model_tot = FitFunc(func_type = 'None',
                        prefix = 'Tot',
                        color = 'lightblue')
    
    if background is not None:
        if background.model is not None:
            model_tot.components[background.prefix] = background
            #check if it is a Shirley sobstitute with a numeric model
            if background.func_type in ['Shirley']:
                background = create_numeric_bkg(y, background)
                # just to create the empty lmfit.Parameters object
                background.create_params()
            #initialise model tot with bkg
            model_tot.model = background.model
            model_tot.params = background.params


            # and sum with all the components
            for c in components:
                if c.model is not None:
                    model_tot.model = model_tot.model + c.model
                    model_tot.params = model_tot.params + c.params
                    model_tot.components[c.prefix] = c
                    
    # if background is not define initialise with the first component
    elif components[0].model is not None:
        model_tot.model = components[0].model
        model_tot.params = components[0].params
        model_tot.components[components[0].prefix] = components[0]
        # and sum with all the components
        for c in components[1:]:
            if c.model is not None:
                model_tot.model = model_tot.model + c.model
                model_tot.params = model_tot.params + c.params
                model_tot.components[c.prefix] = c

    else:
        raise TypeError('not possible to create a composite model with only convolution or Fermi; input at least a components or a background')
    
    if Fermi is not None:
        if Fermi.model is not None:
            model_tot.model = model_tot.model * Fermi.model
            model_tot.params = model_tot.params + Fermi.params
            model_tot.components[Fermi.prefix] = Fermi


    # apply convolution
    if Convolution is not None:
        if Convolution.model is not None:

            from lmfit import CompositeModel

            if Convolution.func_type == 'Gauss_convolution':
                model_tot.model = CompositeModel(model_tot.model, Convolution.model, convolve_gauss)
                model_tot.params = model_tot.params+Convolution.params
                model_tot.components[Convolution.prefix] = Convolution

    return model_tot

def fit_result2Dataset(model_tot, fit_result, x, y):
    '''
    covert lmfit fit_result to a xarray Dataset used to save the fir result
    '''

    ### initialise a dataset (res) where to store the raw data + components

    #check coordinates
    for coord in y.coords:
        if coord != x.name:
            # this auxiliary coordinate can be useful ti save multiDC fitting
            aux = y.coords[coord]

    try:
        res = xr.Dataset(coords={x.name: x, aux.name:aux.data})
    except:
        pass
        res = xr.Dataset(coords={x.name: x})

    # pass the raw data to the dataset with its attributes
    # GENERAL: (type(attribute) must be number, string, list/tuple or narray) if type is different convert to string
    res["DC"] = ((x.name), y.data)
    res['indipendent_vars'] = ((x.name), x.data)
    for key, val in y.attrs.items():
        if not any(isinstance(val, format) for format in [list, str, tuple, np.ndarray,float,int]):
            res["DC"].attrs[key] = str(val)
        else:
            res["DC"].attrs[key] = val

    # pass the total fit curve to the dataset with chi**2 and all the other statisitcal parameters as attributes
    res["fit"] = ((x.name), list(fit_result.best_fit))
    res["fit"].attrs['chi-square'] = fit_result.chisqr
    res["fit"].attrs['reduced chi-square'] = fit_result.redchi
    res["fit"].attrs['errorbars'] = str(fit_result.errorbars)
    res["fit"].attrs['data'] = fit_result.ndata
    res["fit"].attrs['function evals'] = fit_result.nfev
    res["fit"].attrs['variables'] = fit_result.nvarys

    # pass residual
    res["residual"] = ((x.name), fit_result.residual)

    comp_dict = fit_result.eval_components()

    # pass each component
    for prefix in comp_dict.keys():
        #if convolution
        if prefix in ["Conv"]:
            #pass the best fit
            res[prefix] = ((x.name), list(fit_result.best_fit))

        #all the others components
        else:
            res[prefix] = ((x.name), comp_dict[prefix])

        # pass model type, index and prefix as attributes
        res[prefix].attrs['func_type'] = model_tot.components[prefix].func_type
        res[prefix].attrs['prefix'] = prefix

        # pass parameters values, hold, min,max and constranint as attributes
        for param_key in model_tot.params.keys():
            if prefix in param_key:
                res[prefix].attrs[param_key+'_value'] = str(model_tot.params[param_key].value)
                res[prefix].attrs[param_key+'_vary'] = str(model_tot.params[param_key].vary)
                res[prefix].attrs[param_key+'_min'] = str(model_tot.params[param_key].min)
                res[prefix].attrs[param_key+'_max'] = str(model_tot.params[param_key].max)
                res[prefix].attrs[param_key+'_expr'] = str(model_tot.params[param_key].expr)

    return res

def save_fit(model_tot, fit_result, x, y, filename='lastfit.nc'):
    """
    save the fit result in a netCDF4 file (.nc) as a xr.Dataset
    Args:
        model_tot: model uesd in the fit
        fit_result: result of the fit
        x: independent variable as xarray coordinate
        y: xarray conatining the fitted raw data
        filename: file name (extension = .nc)

    Returns:

    """

    # create the xr.dataset to save
    res = fit_result2Dataset(model_tot, fit_result, x, y)
    # save to file
    try:
        res.to_netcdf(filename)
        print('file saved as:',filename)
    except Exception as e:
        print('file NOT saved',e)
        xr.open_dataset(filename).close()
        print('try to save again')


def comp_from_netCDF(dataset):
    "from a dataset create the dictionary of the components"
    # comp_dict is a list containing all the components of the model
    comp_dict = {}
    # for each component in the saved dataset restore the appropriate component in components_dict
    for key,component in dataset.items():
        if key not in ['DC','fit','residual', 'indipendent_vars']:
            temp = FitFunc(dataset[key].func_type, component.prefix)
            temp.create_model()
            temp.eval = dataset[key].data
            for attr in component.attrs:
                if attr not in ['func_type','prefix']:
                    key_par,tag=attr.split('_')
                    if tag == 'value':
                        temp.params[key_par].value = float(component.attrs[attr])
                    if tag == 'vary':
                        if type(component.attrs[attr]) is str:
                            temp.params[key_par].vary = eval(component.attrs[attr])
                        else:
                            temp.params[key_par].vary = component.attrs[attr]
                    if tag == 'min':
                        temp.params[key_par].min = float(component.attrs[attr])
                    if tag == 'max':
                        temp.params[key_par].max = float(component.attrs[attr])
                    if tag == 'expr':
                        if 'None' not in component.attrs[attr]:
                            temp.constr[key_par] = str(component.attrs[attr])

            # append the peak component in the new dictionary

            comp_dict[component.prefix] = temp
    return comp_dict

def load_fit(filename = 'lastfit.nc'):
    '''
    :param filename: name of the file where you saved the lmfit.fit result using the function "save_fit"
    :return:
        comp_dict: dictionary of FitFunc, one for each component (peaks, fermi, background, convolution....)
        ds: xr.array dataset containing all the saved information
    '''

    #open the saved dataset
    with xr.load_dataset(filename) as ds:
        #extract the list of the components
        comp_dict = comp_from_netCDF(ds)

        ds.close()

    # restore the total model by recomposing from comp_dict

    # extract components from dataset
    comp_list = []
    for item_key in comp_dict.keys():
        if item_key.startswith('p'):
            comp_list.append(comp_dict[item_key])

    model_tot = compose_model(y = ds['DC'].data,
                              components = comp_list,
                              background = comp_dict.get('Bkg'),
                              Fermi = comp_dict.get('Fermi'),
                              Convolution = comp_dict.get('Conv'))
    return  model_tot, ds
