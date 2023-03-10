# Denoising: Deep learning-based statistical noise reduction for multidimensional spectral data
# Kim et al. Rev. Sci. Instrum. 92, 073901 (2021)
# https://github.com/Younsik-phys/Denoising-neural-network-for-2D-ARPES-data
# Wrapped for compatibility with peaks by PK 25/10/2021

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os
import cv2
import math
from tqdm.notebook import tqdm
from peaks.utils.metadata import update_hist
from peaks.utils.OOP_method import add_methods


# define neural network. Used skip connection for smooth training a deep neural network.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding = 1, padding_mode = 'replicate', stride = 2) # input channels, output channels, kernel size
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv3 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv4 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv5 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv6 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv7 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv8 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv9 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv10 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv11 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv12 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv13 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv14 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')        
        self.conv15 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv16 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv17 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv18 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv19 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv20 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv21 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv22 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv23 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv24 = nn.Conv2d(64, 64, 3, padding = 1, padding_mode = 'replicate')
        self.conv25 = nn.Conv2d(64, 1, 3, padding = 1, padding_mode = 'replicate')

        self.pool = nn.AvgPool2d(2)
 

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.prelu6 = nn.PReLU()
        self.prelu7 = nn.PReLU()
        self.prelu8 = nn.PReLU()
        self.prelu9 = nn.PReLU()
        self.prelu10 = nn.PReLU()
        self.prelu11 = nn.PReLU()
        self.prelu12 = nn.PReLU()
        self.prelu13 = nn.PReLU()
        self.prelu14 = nn.PReLU()
        self.prelu15 = nn.PReLU()
        self.prelu16 = nn.PReLU()
        self.prelu17 = nn.PReLU()
        self.prelu18 = nn.PReLU()
        self.prelu19 = nn.PReLU()
        self.prelu20 = nn.PReLU()
        self.prelu21 = nn.PReLU()
        self.prelu22 = nn.PReLU()
        self.prelu23 = nn.PReLU()
        self.prelu24 = nn.PReLU()


        self.upsample = nn.Upsample(scale_factor = 2, align_corners = False, mode = 'bicubic')
  

    def forward(self, x):

        x = self.prelu1((self.conv1(x)))
        x = self.prelu2((self.conv2(x)))
        x = self.prelu3((self.conv3(x)))
        x = self.prelu4((self.conv4(x)))
        x = self.prelu5((self.conv5(x)))

        xtemp1 = x

        x = self.prelu6((self.conv6(x)))
        x = self.prelu7((self.conv7(x)))
        x = self.prelu8((self.conv8(x)))
        x = self.prelu9((self.conv9(x)))
        x = self.prelu10((self.conv10(x)))
        
        xtemp2 = x
        x = (x + xtemp1) / 2

        x = self.prelu11((self.conv11(x)))
        x = self.prelu12((self.conv12(x)))
        x = self.prelu13((self.conv13(x)))
        x = self.prelu14((self.conv14(x)))
        x = self.prelu15((self.conv15(x)))

        xtemp1 = x
        x = (x + xtemp2) / 2

        x = self.prelu16((self.conv16(x)))
        x = self.prelu17((self.conv17(x)))
        x = self.prelu18((self.conv18(x)))
        x = self.prelu19((self.conv19(x)))
        x = self.prelu20((self.conv20(x)))

        x = (x + xtemp1) / 2

        x = self.prelu21((self.conv21(x)))
        x = self.prelu22((self.conv22(x)))
        x = self.prelu23((self.conv23(x)))
        x = self.prelu24((self.conv24(x)))

        x = F.relu((self.conv25(x)))

        return x

@add_methods(xr.DataArray)
def denoise(data, nx = 300, ny = 300, full=False, **kwargs):
    ''' This function calls a de-noising function from "Deep learning-based statistical noise reduction for
    multidimensional spectral data", Kim et al. Rev. Sci. Instrum. 92, 073901 (2021), parsing the supplied data
    into 2d arrays for the de-noising

        Input:
            data - the data (single 2D dataarray, map or higher-D dataset (xarray)
            nx (optional, default: 300) - number of points in x (neural network size?) (int)
            ny (optional, default: 300) - number of points in y (neural network size?) (int)
            full (optional, default: False) - if True, overrides nx and ny to use same number as original pixels (bool)
            **kwargs - optional arguments:
                ax - specify (as a single string or a list of strings) the axes to iterate over for denoising
                higher-D datasets. Default behaviour is that de-noising is applied on the analyser detector
                readout data (i.e. slices of theta_par vs. eV for each value of the higher-D dataset), but
                alternative configurations can be specified:
                    e.g. supplying ax = 'eV' for a 3D Fermi surface map dataset will mean that denoising is
                      applied to each constant energy slice
                    e.g. supplying ax = ['eV', 'theta_par'] for a 4D spatial map will mean that denoising is
                      applied to spatial maps taken for each eV, theta_par pixel

        Returns:
            data_out - the de-noised data (xarray) with same shape as orignal input. '''

    # Duplicate data for output
    data_out = data.copy(deep=True)

    # Determine data dimensions
    ndim = len(data.shape)

    if ndim == 2:  # Simplest case, 2D data can be passed directly to the denoising algorithm
        if full==True:  # Called to denoise using the full pixel grid
            nx = data.shape[0]
            ny = data.shape[1]

        data_out.data = denoise2d(data, nx, ny)

    elif ndim == 3:  # 3D dataset supplied
        if 'ax' not in kwargs:
            for count, i in enumerate(data.dims):
                if i != 'eV' and i != 'theta_par':
                    dim_name = i
                    dim_i = count
        else:
            for count, i in enumerate(data.dims):
                if i == kwargs['ax']:
                    dim_name = i
                    dim_i = count

        if full==True:  # Called to denoise using the full pixel grid
            data_shape = data.shape  # Full data shape
            data_shape = [x for x in data_shape if x != data_shape[dim_i]]  # Remove the interating index
            nx = data_shape[0]
            ny = data_shape[1]

        for count, i in enumerate(tqdm(data[dim_name])):
            denoised_data = data[{dim_name: count}].denoise(nx,ny)
            if dim_i == 0:
                data_out[count, :, :] = denoised_data
            elif dim_i == 1:
                data_out[:, count, :] = denoised_data
            elif dim_i == 2:
                data_out[:, :, count] = denoised_data

    elif ndim == 4:  # 4D dataset supplied
        dim_name = []
        dim_i = []
        if 'ax' not in kwargs:
            for count, i in enumerate(data.dims):
                if i != 'eV' and i != 'theta_par':
                    dim_name.append(i)
                    dim_i.append(count)
        else:
            for count, i in enumerate(data.dims):
                if i in kwargs['ax']:
                    dim_name.append(i)
                    dim_i.append(count)

        if full==True:  # Called to denoise using the full pixel grid
            data_shape = data.shape  # Full data shape
            data_shape = [x for x in data_shape if x != data_shape[dim_i[0]] and x != data_shape[dim_i[1]]]  # Remove the interating index
            nx = data_shape[0]
            ny = data_shape[1]

        for count0, i in enumerate(tqdm(data[dim_name[0]], leave=False, desc=str(dim_name[0])+': ')):
            for count1, i2 in enumerate(tqdm(data[dim_name[1]], leave=False, desc=str(dim_name[1])+': ')):
                denoised_data = data[{dim_name[0]: count0, dim_name[1]: count1}].denoise(nx,ny)
                if dim_i[0] == 0:
                    if dim_i[1] == 1:
                        data_out[count0, count1, :, :] = denoised_data
                    elif dim_i[1] == 2:
                        data_out[count0, :, count1, :] = denoised_data
                    elif dim_i[1] == 3:
                        data_out[count0, :, :, count1] = denoised_data
                elif dim_i[0] == 1:
                    if dim_i[1] == 0:
                        data_out[count1, count0, :, :] = denoised_data
                    elif dim_i[1] == 2:
                        data_out[:, count0, count1, :] = denoised_data
                    elif dim_i[1] == 3:
                        data_out[:, count0, :, count1] = denoised_data
                elif dim_i[0] == 2:
                    if dim_i[1] == 0:
                        data_out[count1, :, count0, :] = denoised_data
                    elif dim_i[1] == 1:
                        data_out[:, count1, count0, :] = denoised_data
                    elif dim_i[1] == 3:
                        data_out[:, :, count0, count1] = denoised_data
                elif dim_i[0] == 3:
                    if dim_i[1] == 0:
                        data_out[count1, :, :, count0] = denoised_data
                    elif dim_i[1] == 1:
                        data_out[:, count1, :, count0] = denoised_data
                    elif dim_i[1] == 2:
                        data_out[:, :, count1, count0] = denoised_data

    # Update the analysis history
    hist = 'Denoised using deep learning statistical analysis, (n_x,n_y)=('+str(nx)+','+str(ny)+')'
    if ndim==3:
        hist += ', iterated along axis ' + str(dim_name)
    elif ndim==4:
        hist += ', iterated along axes ' + str(dim_name)
    update_hist(data_out, hist)

    return data_out


def denoise2d(data, nx = 300, ny = 300):
    ''' This function is a de-noising function from "Deep learning-based statistical noise reduction for
    multidimensional spectral data", Kim et al. Rev. Sci. Instrum. 92, 073901 (2021)
    https://github.com/Younsik-phys/Denoising-neural-network-for-2D-ARPES-data

        Input:
            data - the data to be denoised (single 2D numpy array)
            nx (optional, default: 300) - number of points in x (neural network size?)
            ny (optional, default: 300) - number of points in y (neural network size?)

        Returns:
            data_out - the de-noised data (2d numpy array) with same shape and size as orignal input. '''

    inx = data.shape[0]
    iny = data.shape[1]

    data = data.fillna(0)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # if you have memory problem, please use CPU.
    net = Net().to(device)

    net.load_state_dict(torch.load(os.path.dirname(os.path.realpath(__file__)) + "/model91.pt", map_location=device))

    net.eval()

    inputs = data.data
    inputs = cv2.resize(inputs, dsize=(nx, ny), interpolation=cv2.INTER_CUBIC) #image resize
    inputs = inputs / inputs.sum() * 10000 #image normalization

    # 2-D grid has mirror and C4 rotational symmetry. So in total 8 configurations are possible. Input data is converted to 8 configs and they go into the neural network.

    inputs1 = inputs
    inputs2 = np.flip(inputs, axis = 0).copy()
    inputs3 = np.flip(inputs, axis = 1).copy()
    inputs4 = np.flip(np.flip(inputs, axis = 0), axis = 1).copy()

    inputst = np.transpose(inputs)

    inputs5 = inputst
    inputs6 = np.flip(inputst, axis = 0).copy()
    inputs7 = np.flip(inputst, axis = 1).copy()
    inputs8 = np.flip(np.flip(inputst, axis = 0), axis = 1).copy()

    inputs1 = torch.tensor(inputs1).unsqueeze(0).unsqueeze(0).to(device)
    inputs2 = torch.tensor(inputs2).unsqueeze(0).unsqueeze(0).to(device)
    inputs3 = torch.tensor(inputs3).unsqueeze(0).unsqueeze(0).to(device)
    inputs4 = torch.tensor(inputs4).unsqueeze(0).unsqueeze(0).to(device)
    inputs5 = torch.tensor(inputs5).unsqueeze(0).unsqueeze(0).to(device)
    inputs6 = torch.tensor(inputs6).unsqueeze(0).unsqueeze(0).to(device)
    inputs7 = torch.tensor(inputs7).unsqueeze(0).unsqueeze(0).to(device)
    inputs8 = torch.tensor(inputs8).unsqueeze(0).unsqueeze(0).to(device)

    outputs1 = net(inputs1.float().to(device))
    outputs2 = net(inputs2.float().to(device))
    outputs3 = net(inputs3.float().to(device))
    outputs4 = net(inputs4.float().to(device))
    outputs5 = net(inputs5.float().to(device))
    outputs6 = net(inputs6.float().to(device))
    outputs7 = net(inputs7.float().to(device))
    outputs8 = net(inputs8.float().to(device))

    outputs1 = outputs1.to("cpu").squeeze(0).squeeze(0).detach().numpy()
    outputs2 = outputs2.to("cpu").squeeze(0).squeeze(0).detach().numpy()
    outputs3 = outputs3.to("cpu").squeeze(0).squeeze(0).detach().numpy()
    outputs4 = outputs4.to("cpu").squeeze(0).squeeze(0).detach().numpy()
    outputs5 = outputs5.to("cpu").squeeze(0).squeeze(0).detach().numpy()
    outputs6 = outputs6.to("cpu").squeeze(0).squeeze(0).detach().numpy()
    outputs7 = outputs7.to("cpu").squeeze(0).squeeze(0).detach().numpy()
    outputs8 = outputs8.to("cpu").squeeze(0).squeeze(0).detach().numpy()



    outputs1 = outputs1
    outputs2 = np.flip(outputs2, axis = 0)
    outputs3 = np.flip(outputs3, axis = 1)
    outputs4 = np.flip(np.flip(outputs4, axis = 1), axis = 0)

    outputs5 = np.transpose(outputs5)
    outputs6 = np.transpose(np.flip(outputs6, axis = 0))
    outputs7 = np.transpose(np.flip(outputs7, axis = 1))
    outputs8 = np.transpose(np.flip(np.flip(outputs8, axis = 1), axis = 0))

    output = np.zeros((math.ceil(inputs.shape[1]/2), math.ceil(inputs.shape[0]/2), 8))

    output[:, :, 0] = np.flip(outputs1.T, axis = 0)
    output[:, :, 1] = np.flip(outputs2.T, axis = 0)
    output[:, :, 2] = np.flip(outputs3.T, axis = 0)
    output[:, :, 3] = np.flip(outputs4.T, axis = 0)
    output[:, :, 4] = np.flip(outputs5.T, axis = 0)
    output[:, :, 5] = np.flip(outputs6.T, axis = 0)
    output[:, :, 6] = np.flip(outputs7.T, axis = 0)
    output[:, :, 7] = np.flip(outputs8.T, axis = 0)

    outputsavg = output.sum(axis = 2) / 8

    # Re-interpolate back to original image size
    outputs = cv2.resize(outputsavg, dsize=(inx, iny), interpolation=cv2.INTER_CUBIC)

    # Take care of sign reverals
    data_out = np.flip(outputs.T, axis=1)

    return data_out


