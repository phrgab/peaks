Current PyPhoto data loader functionalities


Instructions:

Use the function load_data in data_loading.py to load the data of the file being scanned (avaliable scan types are listed below)
Use the function load_metadata in metadata_loading.py to load metadata from a logbook into a PyPhoto xarray
Use the function save_data in data_saving.py to save an xarray as a netCDF file
Examples of how to load and save data are shown in the 'Data loading example' Jupyter notebook file saved in the examples/fileIO folder


Current compatible scan types:

#St Andrews Phoibos (ARPES system):
Dispersion (limited testing)
FS map (limited testing)
XPS (limited testing)

#St Andrews MBS (Spin-ARPES system):
Dispersion (limited testing - txt file only)

#Diamond i05-HR
Dispersion (limited testing)
FS map (limited testing)
Focus scan (limited testing)
Spatial map (limited testing)
Line scan (limited testing)
Temp dep (limited testing)

#Diamond i05-nano:
Dispersion (limited testing)
FS map (limited testing)
Focus scan (limited testing)
Spatial map (limited testing)
Line scan (limited testing)

#MAX IV Bloch:
Dispersion (limited testing)
FS map (limited testing)

#Elettra APE:
Dispersion (limited testing)
FS map (limited testing)

#SOLEIL CASSIOPEE:
Dispersion (limited testing)
FS map (limited testing)

#Igor Pro waves:
Note: waves must be saved in Igor Pro as follows:
Click Data > Save Waves > Save Igor Text...
Select the wave that you want to save and then click Do It
Change Save as type: from 'IGOR Text Files' to 'Plain Text Files' and then click save
Note: if the loader cannot guess the scan type, an additional argument, e.g. scan_type='XPS', can be included when load_data is called, e.g. load_data(file,scan_type='XPS')
Dispersion (limited testing)
XPS (limited testing)
EDCs/MDCs (limited testing)

#netCDF:
xarrays saved as netCDF can be loaded
netCDF files can be loaded as xarrays
