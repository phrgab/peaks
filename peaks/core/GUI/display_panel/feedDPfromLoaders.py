# Feed data from peaks loaders to the Display panel
# Edgar Abarca Morales

# Create a Display Panel input dictionary from peaks loaders
def feedDPfromLoaders(a):

    # If data is 2D
    if len(a.dims) == 2:

        aT = a.T

        # Extract (x,y) dimensions
        xdim=aT.dims[0]
        ydim=aT.dims[1]

        # Define z dimension
        zdim='counts'

        # Extract x units
        if 'units' in aT.coords[xdim].attrs:
            xuts=aT.coords[xdim].attrs['units']
        else:
            xuts=''

        # Extract y units
        if 'units' in aT.coords[ydim].attrs:
            yuts=aT.coords[ydim].attrs['units']
        else:
            yuts=''

        # Define z units
        zuts=None

        # Extract data
        x=aT.coords[xdim].data
        y=aT.coords[ydim].data
        z=aT.data

        DP_dict={'x':x,'y':y,'z':z,'xdim':xdim,'ydim':ydim,'zdim':zdim,'xuts':xuts,'yuts':yuts,'zuts':zuts}

    return DP_dict
