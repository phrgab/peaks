# Feed data from peaks loaders to the Display panel
# Edgar Abarca Morales

# Create a Display Panel input dictionary from peaks loaders
def feedDPfromLoaders(a):

    # If data is 2D
    if len(a.dims) == 2:

        # Extract (x,y) dimensions
        xdim=a.dims[0]
        ydim=a.dims[1]

        # Define z dimension
        zdim='counts'

        # Extract x units
        if 'units' in a.coords[xdim].attrs:
            xuts=a.coords[xdim].attrs['units']
        else:
            xuts=''

        # Extract y units
        if 'units' in a.coords[ydim].attrs:
            yuts=a.coords[ydim].attrs['units']
        else:
            yuts=''

        # Define z units
        zuts=None

        # Extract data
        x=a.coords[xdim].data
        y=a.coords[ydim].data
        z=a.data

        DP_dict={'x':x,'y':y,'z':z,'xdim':xdim,'ydim':ydim,'zdim':zdim,'xuts':xuts,'yuts':yuts,'zuts':zuts}

        return DP_dict
