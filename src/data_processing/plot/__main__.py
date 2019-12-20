#!/usr/bin/python
import sys, os

sys.path.append('/home/karim/workspace/vscode-python/ADNI_Data_processing/src/data_processing')

import config.config_read as rsd
import services.tools as tls
import io_data.data_acces_file as daf
import matplotlib.pyplot as plt 
import numpy as np

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

    sag_L, sag_R = get_sag_slices(data_roi_L, data_roi_R, sag_l, sag_r)
    cor_L, cor_R = get_cor_slices(data_roi_L, data_roi_R, cor_l, cor_r)
    axi_L, axi_R = get_axi_slices(data_roi_L, data_roi_R, axi_l, axi_r)
        
    # plot 2D slice from ROI (m-1, m, m+1) 
    for i in range(3):        
        plt.subplot(3, 6, i+1)       
        plt.imshow(sag_L[i, :, :], cmap='gray', origin="lower")        
        plt.subplot(3, 6, 4+i)       
        plt.imshow(sag_R[i, :, :], cmap='gray', origin="lower") 
               
        plt.subplot(3, 6, 6+i+1)               
        plt.imshow(cor_L[:, i, :], cmap='gray', origin="lower")        
        plt.subplot(3, 6, 6+4+i)       
        plt.imshow(cor_R[:, i, :], cmap='gray', origin="lower")
        
        plt.subplot(3, 6, 12+i+1)       
        plt.imshow(axi_L[:, :, i], cmap='gray', origin="lower")
        plt.subplot(3, 6, 12+4+i)       
        plt.imshow(axi_R[:, :, i], cmap='gray', origin="lower")  
    plt.show()
        
def get_pickle_from_folder(folder):
    res = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('pkl'): 
                res.append(os.path.join(root, file))
    return res
  
        
#------------------------------------------------------------------------------------------
# function::__main__ ::
#------------------------------------------------------------------------------------------
def main():
    binaries_classes = ['AD-NC', 'AD-MCI', 'MCI-NC']
    data_params = rsd.get_all_data_params()
    root_path = data_params['adni_data_des']
    name_cnv = root_path + tls.get_convention_name(data_params) + '/' + str(data_params['ROI_list'][data_params['ROI_selection']] + '/' + data_params['3D_or_2D'])
    line = name_cnv + '/' + binaries_classes[0] + '/test/'
    list_files = get_pickle_from_folder(line)

    for i in list_files:
        model = daf.read_model(i)
        print(" HIPP_L : {} - HIPP_R: {} - Vector: {} - Label: {}".format(model.hippLeft.shape, model.hippRight.shape, model.hippMetaDataVector, model.hippLabel))
        # print(model)
        left_dims, right_dims = [[13,16],[13,16],[13,16]], [[13,16],[13,16],[13,16]]
        plot_ROI_all(model.hippLeft, model.hippRight, left_dims, right_dims)
   


#------------------------------------------------------------------------------------------
# Start ->>>->>>  
#------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
