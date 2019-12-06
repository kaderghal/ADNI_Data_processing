#!/usr/bin/python

import config.config_read as rsd
import io_data.xml_acces_file as xaf
from models.Subject import Subject
import os
import errno
import numpy as np
import nibabel as nib
import numpy.random as rnd
import math
import xml.etree.ElementTree as ET
from random import shuffle

#------------------------------------------------------------------------------------------
# get Subject Info [Age, Sex, MMSE]
#------------------------------------------------------------------------------------------
def get_meta_data_xml(data_params, subject_ID):
    return xaf.get_Subject_info(data_params, subject_ID)    
          
#------------------------------------------------------------------------------------------
# Function generates a conevntionned name from the config parameters
# -> Example : if factor is 100, dim is 28, max shift is 2, and gaussian blurr is 1.2
# the name will be  => F_28P_F100_MS02_MB12D
#------------------------------------------------------------------------------------------
def get_convention_name(data_params):
    dim_pixel = get_dimensions_patch(data_params)[0]
    path_file = 'F_' + str(dim_pixel) + 'P_F' + str(data_params['factor']) + '_MS' + str(data_params['shift']) + '_MB' + str(data_params['sigma']).replace(".", "") + 'D'
    return path_file

#------------------------------------------------------------------------------------------
# Function comptues the rotation operation
#------------------------------------------------------------------------------------------
def matrix_rotation(mtx):
    transpose = np.transpose(np.transpose(mtx)) 
    # rotated = list(reversed(zip(*transpose))) # for python 2
    rotated = list(zip(*transpose))[::-1] # for python 3
    return rotated

#------------------------------------------------------------------------------------------
# Function comptues the rotation operation for 3D volume
#------------------------------------------------------------------------------------------
def volume_rotation(data):
    data_holder = np.zeros((data.shape[0], data.shape[1], data.shape[2]))   
    for i in range(data.shape[0]): # sag dimension
        data_holder[i, :, :] = list(zip(*np.transpose(np.transpose(data[i, :, :]))))[::-1]
    return data_holder
     
#------------------------------------------------------------------------------------------
# Function create folder if not exits 
#------------------------------------------------------------------------------------------
def create_folder(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

#------------------------------------------------------------------------------------------
# Function return numpy array data from Nifty file 
#------------------------------------------------------------------------------------------
def nii_to_array(nii_filename, data_type, fix_nan=True):  # get 3d array from the nii image
    img = nib.load(nii_filename)
    np_data = img.get_data().astype(data_type)
    if fix_nan:
        np_data = np.nan_to_num(np_data)
    return np_data

#------------------------------------------------------------------------------------------
# Function generate list of data: for a selected modality and binary classification
# -> Example: if label=AD-NC and modality=DTI we will get a list like:
# -> []
#------------------------------------------------------------------------------------------
def generate_selected_list(label, modal_type):  # ('NC-AD', 'DTI')
    lst_data = []
    list_data_set = ['alz_sMRI_train', 'alz_sMRI_valid', 'alz_sMRI_test', 'alz_MD_train', 'alz_MD_valid', 'alz_MD_test']
    for item in list_data_set:
        if modal_type in item:
            str_ = item + '_' + label
            lst_data.append(str_)
    return lst_data

#------------------------------------------------------------------------------------------
# Function: 
#------------------------------------------------------------------------------------------
def generate_selected_label_list(label):  # ('NC-AD', '')
    lst_data = []
    list_data_set = ['alz_sMRI_train', 'alz_sMRI_valid', 'alz_sMRI_test', 'alz_MD_train', 'alz_MD_valid', 'alz_MD_test']
    for item in list_data_set:
        str_ = item + '_' + label
        lst_data.append(str_)
    return lst_data

#------------------------------------------------------------------------------------------
# Function: split binary classification groups
#------------------------------------------------------------------------------------------
def split_lists_to_binary_groups(lists):
    three_way_labels = list(rsd.get_label_binary_codes()['AD-MCI-NC'].keys())  # {'AD': 0, 'MCI': 1, 'NC': 2}    
    # bin_labels = {'10': three_way_labels[1] + '-' + three_way_labels[0], '12': three_way_labels[1] + '-' + three_way_labels[2], '20': three_way_labels[2] + '-' + three_way_labels[0]}
    bin_labels = {'01': three_way_labels[0] + '-' + three_way_labels[1], '12': three_way_labels[1] + '-' + three_way_labels[2], '02': three_way_labels[0] + '-' + three_way_labels[2]}
    bin_groups = {'01': [], '12': [], '02': []}    
    for item in lists:
        if item[0] == three_way_labels[0]:
            bin_groups['01'].append(item)
            bin_groups['02'].append(item)
        if item[0] == three_way_labels[1]:
            bin_groups['01'].append(item)
            bin_groups['12'].append(item)
        if item[0] == three_way_labels[2]:
            bin_groups['12'].append(item)
            bin_groups['02'].append(item)
    return {bin_labels[k]: bin_groups[k] for k in ('01', '12', '02')}


## 
def split_mri_dti(item_list):
    mri_list = [(i[0], i[1], i[3]) for i in item_list]
    dti_list = [(i[0], i[2], i[3]) for i in item_list]
    return mri_list, dti_list

##
def get_dimensions_patch(data):
    hipp = data['hipp_left']
    padding_size = int(data['padding_size'])
    shift_param = int(data['shift'])
    new_hipp = (
        int(hipp[0]) - 1 - shift_param - padding_size, int(hipp[1]) - 1 - shift_param + padding_size,
        int(hipp[2]) - 1 - shift_param - padding_size, int(hipp[3]) - 1 - shift_param + padding_size,
        int(hipp[4]) - 1 - shift_param - padding_size, int(hipp[5]) - 1 - shift_param + padding_size)
    return (new_hipp[1] - new_hipp[0], new_hipp[3] - new_hipp[2], new_hipp[5] - new_hipp[4])

#------------------------------------------------------------------------------------------
# generates Augmentation parameters (not randomly) to avoid same params
# -> version 1: overfitting !!!! 
#------------------------------------------------------------------------------------------

def generate_augm_params(max_blur, max_shift):
    import numpy.random as rnd
    while True:
        shift_x = rnd.randint(-max_shift, max_shift + 1)
        shift_y = rnd.randint(-max_shift, max_shift + 1)
        shift_z = rnd.randint(-max_shift, max_shift + 1)
        blur_sigma = float(rnd.randint(1000)) / 1000 * max_blur
        if shift_x + shift_y + shift_z + blur_sigma > 0:  # ???
            return shift_x, shift_y, shift_z, blur_sigma

#------------------------------------------------------------------------------------------
# generates Augmentation parameters (not randomly) to avoid same params
# -> NB: pour eviter : overfitting 
#------------------------------------------------------------------------------------------
def generate_augmentation_parameters_list_v2(max_blur, max_shift):
    res = []
    if max_blur is not None and max_shift is not None:
        L = [round(x*0.100, 2) for x in range(0, (int(max_blur * 10) + 1))]
        for idx in range(len(L)):
            for i in range(-max_shift, max_shift + 1):
                for j in range(-max_shift, max_shift + 1):
                    for k in range(-max_shift, max_shift + 1):
                        res.append([(i, j, k, L[idx])])

    return res

#------------------------------------------------------------------------------------------
# generate parameters (x, y, z, k.k) for each subject
# -> overfitting !!!! 
#------------------------------------------------------------------------------------------

def generate_augm_lists(dirs_with_labels, new_size, max_blur, max_shift, default_augm_params=None):
    # pas d'augmentation
    if new_size is None or len(dirs_with_labels) == new_size:  # ?
        return [dwl + [default_augm_params] for dwl in dirs_with_labels]
    
    # for avoid similar augmentation  
    local_param_list = generate_augmentation_parameters_list_v2(max_blur, max_shift)
    shuffle(local_param_list)
    augm_coeff = int(math.floor(new_size / len(dirs_with_labels)))    
    res = []
    i = 0
    for dwl in dirs_with_labels:
        res.append(dwl + [(0, 0, 0, 0.0)])
        i += 1
        # for _ in range(augm_coeff - 1):
            # print("i : ", i)
            # res.append(dwl + local_param_list[i])
            # i += 1
    #print("-->", dirs_with_labels[0][0], augm_coeff, len(dirs_with_labels), len(res), i)        
    while i < new_size:
        ridx = rnd.randint(len(dirs_with_labels))
        dwl = dirs_with_labels[ridx]
        res.append(dwl + local_param_list[i])
        i += 1
    # for x in res:
    #     print(x[0], x[3])
    return res

#------------------------------------------------------------------------------------------
# generate list dataset with aumentation parameters 
# EXP: ['AD', 'path/to/MRI/', (x,y,z, a.b)] for each subject
#------------------------------------------------------------------------------------------

def generate_augm_lists_v2(dirs_with_labels, new_size, max_blur, max_shift, default_augm_params=None):
    # pas d'augmentation
    if new_size is None or len(dirs_with_labels) == new_size or max_blur == 0 or max_shift == 0:  # ?
        return [dwl + [default_augm_params] for dwl in dirs_with_labels]

    augm_coeff = int(math.floor(new_size / len(dirs_with_labels)))
    res = []
    local_param_list = generate_augmentation_parameters_list_v2(max_blur, max_shift)
    shuffle(local_param_list)            
    for dwl in dirs_with_labels:
        res.append(dwl + [(0, 0, 0, 0.0)])
        for _ in range(augm_coeff - 1):
            res.append(dwl + local_param_list.pop())   
        local_param_list = generate_augmentation_parameters_list_v2(max_blur, max_shift)
        shuffle(local_param_list)
    for _ in range(new_size - len(dirs_with_labels) * augm_coeff):
        ridx = rnd.randint(len(dirs_with_labels))
        dwl = dirs_with_labels[ridx] 
        res.append(dwl + local_param_list.pop())     
    return res


def get_dimensions_cubes_HIPP(data):
    crp_l = data['hipp_left']
    crp_r = data['hipp_right']
    padding_size = int(data['padding_size'])
    shift_param = int(data['shift'])
    #
    #[int(crp_l[i]) - 1 - shift_param - padding_size, int(crp_l[i+1]) - 1 - shift_param + padding_size \
     #   for i in range(0, 6, 2)]
    new_crp_l = (
        int(crp_l[0]) - 1 - shift_param - padding_size, int(crp_l[1]) - 1 - shift_param + padding_size,
        int(crp_l[2]) - 1 - shift_param - padding_size, int(crp_l[3]) - 1 - shift_param + padding_size,
        int(crp_l[4]) - 1 - shift_param - padding_size, int(crp_l[5]) - 1 - shift_param + padding_size)
    new_crp_r = (int(crp_r[0]) - 1 - shift_param - padding_size, int(crp_r[1]) - 1 - shift_param + padding_size,
                 int(crp_r[2]) - 1 - shift_param - padding_size, int(crp_r[3]) - 1 - shift_param + padding_size,
                 int(crp_r[4]) - 1 - shift_param - padding_size, int(crp_r[5]) - 1 - shift_param + padding_size)
    return new_crp_l, new_crp_r


def get_dimensions_cubes_PPC(data):
    crp_l = data['ppc_left']
    crp_r = data['ppc_right']
    padding_size = int(data['padding_size'])
    shift_param = int(data['shift'])
    new_crp_l = (
        int(crp_l[0]) - 1 - shift_param - padding_size, int(crp_l[1]) - 1 - shift_param + padding_size,
        int(crp_l[2]) - 1 - shift_param - padding_size, int(crp_l[3]) - 1 - shift_param + padding_size,
        int(crp_l[4]) - 1 - shift_param - padding_size, int(crp_l[5]) - 1 - shift_param + padding_size)
    new_crp_r = (int(crp_r[0]) - 1 - shift_param - padding_size, int(crp_r[1]) - 1 - shift_param + padding_size,
                 int(crp_r[2]) - 1 - shift_param - padding_size, int(crp_r[3]) - 1 - shift_param + padding_size,
                 int(crp_r[4]) - 1 - shift_param - padding_size, int(crp_r[5]) - 1 - shift_param + padding_size)
    return new_crp_l, new_crp_r

#
def split_list(liste, number_ele):
    return liste[:number_ele], liste[number_ele:]

def getSubjectByID(data_params, subjectID):        
    subjectEle = xaf.get_Subject_info(data_params, subjectID) #[ID, Date, Class, Age, Sex, MMSE] 
    return Subject(*subjectEle)