#!/usr/bin/python

import numpy as np
import services.tools as tls
import matplotlib.pyplot as plt 
from PIL import Image

#------------------------------------------------------------------------------------------
# Plot slices from the selected ROI
#------------------------------------------------------------------------------------------


def get_sag_slices(data_L, data_R, sag_l, sag_r):        
    selected_data_L = data_L[sag_l[0]:sag_l[1], :, :]
    selected_data_R = data_R[sag_r[0]:sag_r[1], :, :]   
    return selected_data_L, selected_data_R

def get_cor_slices(data_L, data_R, cor_l, cor_r):        
    selected_data_L = data_L[:, cor_l[0]:cor_l[1], :]
    selected_data_R = data_R[:, cor_r[0]:cor_r[1], :]           
    return selected_data_L, selected_data_R

def get_axi_slices(data_L, data_R, axi_l, axi_r):        
    selected_data_L = data_L[:, :, axi_l[0]:axi_l[1]]
    selected_data_R = data_R[:, :, axi_r[0]:axi_r[1]]          
    return selected_data_L, selected_data_R




def plot_ROI_all(data_roi_L, data_roi_R, left_dims, right_dims):
    sag_l, cor_l, axi_l = left_dims     
    sag_r, cor_r, axi_r = right_dims
    # sagittal slices
    sag_L, sag_R = get_sag_slices(data_roi_L, data_roi_R, sag_l, sag_r)
    cor_L, cor_R = get_cor_slices(data_roi_L, data_roi_R, cor_l, cor_r)
    axi_L, axi_R = get_axi_slices(data_roi_L, data_roi_R, axi_l, axi_r)
    
    # """ Function to display row of image slices """
    # fig, axes = plt.subplots(1, len(slices))
    # for i, slice in enumerate(slices):
    #     axes[i].imshow(slice.T, cmap="gray", origin="lower")
        
   
        
    # plot 2D slice from ROI (m-1, m, m+1) 
    for i in range(3):        
        plt.subplot(3, 6, i+1)       
        plt.imshow(sag_L[i, :, :], cmap='gray')        
        plt.subplot(3, 6, 4+i)       
        plt.imshow(sag_R[i, :, :], cmap='gray') 
               
        plt.subplot(3, 6, 6+i+1)               
        plt.imshow(cor_L[:, i, :], cmap='gray')        
        plt.subplot(3, 6, 6+4+i)       
        plt.imshow(cor_R[:, i, :], cmap='gray')
        
        plt.subplot(3, 6, 12+i+1)       
        plt.imshow(axi_L[:, :, i], cmap='gray')
        plt.subplot(3, 6, 12+4+i)       
        plt.imshow(axi_R[:, :, i], cmap='gray')  
    plt.show()