#Functions to save an xarray as a netCDF file
#Brendan Edwards 16/02/2021

from ipywidgets import Layout, widgets
from IPython.display import display, clear_output
from ipyfilechooser import FileChooser
from peaks.core.utils.OOP_method import add_methods
import xarray as xr

@add_methods(xr.DataArray)
def save_data(xarray, file):
    '''This function saves an xarray as a netCDF file
    
    Input:
        xarray - The xarray being saved as a netCDF file (xarray)
        file - Path to the file beig created (string)'''
    
    #copy the input xarray
    netCDF_file = xarray.copy(deep=True)
        
    #netCDF struggles to save lists
    analysis_history = ''
    for item in netCDF_file.attrs['analysis_history']:
        analysis_history = analysis_history + ',' + item
    netCDF_file.attrs['analysis_history'] = analysis_history
    
    #netCDF cannot save None type
    for attr in netCDF_file.attrs:
        if netCDF_file.attrs[attr] == None:
            netCDF_file.attrs[attr] = ''
            
    #save xarray as netCDF
    netCDF_file.to_netcdf(file)

class save(object):
    '''This class acts as a UI for saving data using peaks'''
    
    def __init__(self, xarray):
        self.xarray = xarray
        self.display_widgets()

    def _create_widgets(self):
        self.choose_folder = FileChooser('../../')
        self.choose_folder.show_only_dirs = True
        self.file_name = widgets.Text(
            placeholder='Please enter the file name',
            description='File name:',
            style = {'description_width': 'initial'},
            layout = Layout(width = '249px'),
            disabled=False
        )   
        self.saveButton = widgets.Button(description="Save as netCDF",
                                     tooltip='Click to save seleted files',
                                     button_style='success',
                                     layout = Layout(width = '145px'),
                                     )
        self.saveButton.on_click(self._on_saveButton_clicked)
        self.out = widgets.Output(layout={'border': '1px solid black', 'width' : '400px'})
    
    def _on_saveButton_clicked(self, change):
        if self.file_name.value == '':
            with self.out:
                clear_output()
                print('Please enter a file name')
            return
        file_path = str(self.choose_folder.selected_path) + '\\' + self.file_name.value + '.nc'
        with self.out:
            clear_output()
            print('Saving xarray as a netCDF file')
        try:
            save_data(self.xarray, file_path)
            with self.out:
                print(self.file_name.value + '.nc saved')
        except FileNotFoundError:
            with self.out:
                print('Saving failed. Please select a folder')

    #display UI
    def display_widgets(self):
        self._create_widgets()
        with self.out:
            print('To save an xarray, select a folder, enter the file \nname and click \"Save as netCDF\"')
        #Display panel
        display(widgets.VBox([self.choose_folder,widgets.HBox([self.file_name, self.saveButton]), self.out]))
