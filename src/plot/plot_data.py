

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

