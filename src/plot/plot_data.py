

from PIL import Image
import numpy as np
import services.tools as tls
import matplotlib.pyplot as plt 


###############################################################################################################################
# Plot slices from ROI HIPP
#
###############################################################################################################################

def plot_HIPP(data_roi, projection, slc_index_begin, slc_index_end):    
    if projection == 0: # sag
        selected_data = data_roi[slc_index_begin:slc_index_end, :, :]
        data = np.transpose(selected_data, (0, 1, 2)) # 3,28,28
    elif projection == 1: # cor
        data = data_roi[:, slc_index_begin:slc_index_end, :]
        data = np.transpose(data, (1, 0, 2))
    else: # axi
        data = data_roi[:, :, slc_index_begin:slc_index_end]
        data = np.transpose(data, (2, 0, 1))
    # create a container to hold transposed data
    container = np.zeros((data.shape[1], data.shape[2], 3)) # 28,28,3          
    container[:, :, 0], container[:, :, 1], container[:, :, 2] = [np.array(tls.matrix_rotation(data[i, :, :])) for i in range(3)]    
    
    # plot 2D slice from ROI (m-1, m, m+1) 
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(container[:, :, i], cmap='gray')
        
    plt.show()




def get_sag_slices(data_L, data_R, slc_index_begin, slc_index_end):        
    # sagittal slices
    selected_data_L = data_L[slc_index_begin:slc_index_end, :, :]
    sag_data_L = np.transpose(selected_data_L, (0, 1, 2)) # 3,28,28
    selected_data_R = data_R[slc_index_begin:slc_index_end, :, :]
    sag_data_R = np.transpose(selected_data_R, (0, 1, 2)) # 3,28,28  
    container_L = np.zeros((sag_data_L.shape[1], sag_data_L.shape[2], 3)) # 28,28,3  
    container_L[:, :, 0], container_L[:, :, 1], container_L[:, :, 2] = [np.array(tls.matrix_rotation(sag_data_L[i, :, :])) for i in range(3)]  
    container_R = np.zeros((sag_data_R.shape[1], sag_data_R.shape[2], 3)) # 28,28,3  
    container_R[:, :, 0], container_R[:, :, 1], container_R[:, :, 2] = [np.array(tls.matrix_rotation(sag_data_R[i, :, :])) for i in range(3)]       
    return container_L, container_R

def get_cor_slices(data_L, data_R, slc_index_begin, slc_index_end):        
    # sagittal slices
    selected_data_L = data_L[:, slc_index_begin:slc_index_end, :]
    cor_data_L = np.transpose(selected_data_L, (1, 0, 2)) # 3,28,28
    selected_data_R = data_R[:, slc_index_begin:slc_index_end, :]
    cor_data_R = np.transpose(selected_data_R, (1, 0, 2)) # 3,28,28        
    container_L = np.zeros((cor_data_L.shape[1], cor_data_L.shape[2], 3)) # 28,28,3  
    container_L[:, :, 0], container_L[:, :, 1], container_L[:, :, 2] = [np.array(tls.matrix_rotation(cor_data_L[i, :, :])) for i in range(3)]  
    container_R = np.zeros((cor_data_R.shape[1], cor_data_R.shape[2], 3)) # 28,28,3  
    container_R[:, :, 0], container_R[:, :, 1], container_R[:, :, 2] = [np.array(tls.matrix_rotation(cor_data_R[i, :, :])) for i in range(3)]       
    return container_L, container_R


def get_axi_slices(data_L, data_R, slc_index_begin, slc_index_end):        
    # sagittal slices
    selected_data_L = data_L[:, :, slc_index_begin:slc_index_end]
    axi_data_L = np.transpose(selected_data_L, (2, 0, 1)) # 3,28,28
    selected_data_R = data_R[:, :, slc_index_begin:slc_index_end]
    axi_data_R = np.transpose(selected_data_R, (2, 0, 1)) # 3,28,28        
    container_L = np.zeros((axi_data_L.shape[1], axi_data_L.shape[2], 3)) # 28,28,3  
    container_L[:, :, 0], container_L[:, :, 1], container_L[:, :, 2] = [np.array(tls.matrix_rotation(axi_data_L[i, :, :])) for i in range(3)]  
    container_R = np.zeros((axi_data_R.shape[1], axi_data_R.shape[2], 3)) # 28,28,3  
    container_R[:, :, 0], container_R[:, :, 1], container_R[:, :, 2] = [np.array(tls.matrix_rotation(axi_data_R[i, :, :])) for i in range(3)]       
    return container_L, container_R





def plot_ROI_all(data_roi_L, data_roi_R, slc_index_begin, slc_index_end):      
    # sagittal slices
    sag_L, sag_R = get_sag_slices(data_roi_L, data_roi_R, slc_index_begin, slc_index_end)
    cor_L, cor_R = get_cor_slices(data_roi_L, data_roi_R, slc_index_begin, slc_index_end)
    axi_L, axi_R = get_axi_slices(data_roi_L, data_roi_R, slc_index_begin, slc_index_end)

    # plot 2D slice from ROI (m-1, m, m+1) 
    for i in range(3):
        
        plt.subplot(3, 6, i+1)       
        plt.imshow(sag_L[:, :, i], cmap='gray')
        
        plt.subplot(3, 6, 4+i)       
        plt.imshow(sag_R[:, :, i], cmap='gray')
        
        
        plt.subplot(3, 6, 6+i+1)       
        plt.imshow(cor_L[:, :, i], cmap='gray')
        
        plt.subplot(3, 6, 6+4+i)       
        plt.imshow(cor_R[:, :, i], cmap='gray')
        
        
        plt.subplot(3, 6, 12+i+1)       
        plt.imshow(axi_L[:, :, i], cmap='gray')
        
        plt.subplot(3, 6, 12+4+i)       
        plt.imshow(axi_R[:, :, i], cmap='gray')
        
               
        
    plt.show()
    
    