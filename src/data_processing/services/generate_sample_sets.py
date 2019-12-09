#!/usr/bin/python

import os
import sys
import time
import math
import config.config_read as rsd
import config.config_init as cfg
import config.ColorPrompt as CP
import services.tools as tls
import io_data.data_acces_file as daf
import interface.inline_print as iprint
import services.process as prc
import plot.plot_data as plot_data
import plot.plot_3D_ROI as plot_3D_data
from models.HippModel import HippModel
from PIL import Image
from sys import getsizeof
import scipy.misc
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt 

#------------------------------------------------------------------------------------------
# Function: generate_lists() generates and saves list of data of MRI with augmentation
#   (tain, Valid and Test) 
#------------------------------------------------------------------------------------------

def generate_lists(data_params):
    list_data = []
    file_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/List_data.pkl'
    adni_out = generate_lists_from_adni_dataset(data_params)
    list_data.append(adni_out)
    daf.save_lists_to_file(path_file=file_path, data_list=list_data)

#------------------------------------------------------------------------------------------
# Function: split data by using class name (from txt file)
#------------------------------------------------------------------------------------------

def split_classses_data(liste):
    liste_AD = []
    liste_MCI = []
    liste_NC = []
    for item in liste:
        if 'AD' in item.split(':')[1]:
            liste_AD.append(item.split(':')[0])

    for item in liste:
        if 'MCI' in item.split(':')[1]:
            liste_MCI.append(item.split(':')[0])

    for item in liste:
        if 'NC' in item.split(':')[1]:
            liste_NC.append(item.split(':')[0])
    return liste_AD, liste_MCI, liste_NC

#------------------------------------------------------------------------------------------
# Function: generates lists from ADNI folder dataset
#------------------------------------------------------------------------------------------

def get_subjects_with_classes(data_params):
    AD, MCI, NC = split_classses_data(daf.read_data_file(str(data_params['adni_1_classes'])))
    time.sleep(1)
    return [AD, MCI, NC]

#------------------------------------------------------------------------------------------
# Function: generates lists from ADNI folder dataset
#------------------------------------------------------------------------------------------

def generate_lists_from_adni_dataset(data_params, shuffle_data=False, debug=False):
    stage_classes = ['AD', 'MCI', 'NC']
    max_blur = float(data_params['sigma'])
    max_shift = int(data_params['shift'])
    default_augm = (0, 0, 0, 0.0)
    adni1_list = get_subjects_with_classes(data_params)
    adni1_size = {'AD': len(adni1_list[0]), 'MCI': len(adni1_list[1]), 'NC': len(adni1_list[2])}
    adni_1_labels = {'AD': adni1_list[0], 'MCI': adni1_list[1], 'NC': adni1_list[2]}
    adni_1_dirs_root = {k: [data_params['adni_1_brain_data'] + '/' + i for i in adni_1_labels[k]] for k in stage_classes}

    # Test statement for spliting data (check "SPLIT_SET_PARAMS" in config file)
    if(data_params['static_split']):
        test_selected_size = {k: int(data_params['select_test'][k]) for k in stage_classes}
        valid_selected_size = {k: int(data_params['select_valid'][k]) for k in stage_classes} 
    else:
        test_selected_size = {k: int(data_params['select_test'][k]) for k in stage_classes}
        valid_selected_size = {k: int(math.ceil(int(adni1_size[k]) * 20) / 100.0) for k in stage_classes} 
    

    train_selected_size = {k: adni1_size[k] - valid_selected_size[k] - test_selected_size[k] for k in stage_classes}    

    # split checked
    adni_1_test  = {k: adni_1_dirs_root[k][:int(test_selected_size[k])] for k in stage_classes}
    adni_1_valid = {k: adni_1_dirs_root[k][int(test_selected_size[k]):int(test_selected_size[k]) + int(valid_selected_size[k])] for k in stage_classes}
    adni_1_train = {k: adni_1_dirs_root[k][int(test_selected_size[k]) + int(valid_selected_size[k]): int(test_selected_size[k]) + int(valid_selected_size[k]) + int(train_selected_size[k])] for k in stage_classes}

    adni_1_train_size_balanced = int(max(train_selected_size.values()) * int(data_params['factor']))
    adni_1_valid_size_balanced = int(max(valid_selected_size.values()) * int(data_params['factor']))
    
    if data_params['augm_test']:
        adni_1_test_size = int(max(test_selected_size.values()) * int(data_params['factor']))
    else:
        adni_1_test_size = int(min(test_selected_size.values()))

        
    adni_1_train_size_print = adni_1_train_size_balanced
    adni_1_valid_size_print = adni_1_valid_size_balanced
    adni_1_test_size_print  = adni_1_test_size

    if data_params['flip']:
        adni_1_train_size_print = adni_1_train_size_balanced * 2
        adni_1_valid_size_print = adni_1_valid_size_balanced * 2
        adni_1_test_size_print  = adni_1_test_size  * 2

    print('\n--------------------------------------------------------------------------')
    print('* [' + CP.fg.YELLOW + 'train'+ CP.fg.WHITE + '] data will be augmented to {} samples by each class'.format(adni_1_train_size_print))
    print('* [' + CP.fg.YELLOW + 'valid'+ CP.fg.WHITE + '] data will be augmented to {} samples by each class'.format(adni_1_valid_size_print))
    print('* [' + CP.fg.YELLOW + 'test' + CP.fg.WHITE + '] data will be augmented to {} samples by each class'.format(adni_1_test_size_print))
    print('--------------------------------------------------------------------------\n')

    # print table of data augmentation    
    iprint.print_augmentation_table([
    [int(train_selected_size['AD']), int(train_selected_size['MCI']), int(train_selected_size['NC']),  adni_1_train_size_print],
    [int(valid_selected_size['AD']), int(valid_selected_size['MCI']), int(valid_selected_size['NC']),  adni_1_valid_size_print],
    [int(test_selected_size['AD']), int(test_selected_size['MCI']), int(test_selected_size['NC']),  adni_1_test_size_print]])


    adni_1_train_lists_out = []
    adni_1_valid_lists_out = []
    adni_1_test_lists_out = []

    for k in stage_classes:
        adni_1_test_lists =  [[k, i + '/MRI/'] for i in adni_1_test[k]]
        adni_1_valid_lists = [[k, i + '/MRI/'] for i in adni_1_valid[k]]
        adni_1_train_lists = [[k, i + '/MRI/'] for i in adni_1_train[k]]
              
        adni_1_test_lists_out += tls.generate_augm_lists_v2(adni_1_test_lists, None, None, None, default_augm_params=default_augm)
        adni_1_valid_lists_out += tls.generate_augm_lists_v2(adni_1_valid_lists, adni_1_valid_size_balanced, max_blur, max_shift, default_augm_params=default_augm)
        adni_1_train_lists_out += tls.generate_augm_lists_v2(adni_1_train_lists, adni_1_train_size_balanced, max_blur, max_shift, default_augm_params=default_augm)
          
        if shuffle_data:
            rnd.shuffle(adni_1_train_lists_out)
            rnd.shuffle(adni_1_valid_lists_out)

        if debug:
            print ('########################### MRI ##########################')
            print('### train lists (%d instances):' % len(adni_1_train_lists_out))
            for i in adni_1_train_lists_out:
                print(i)
            # ########################
            time.sleep(3)
            # ########################
            print('### valid lists (%d instances):' % len(adni_1_valid_lists_out))
            for i in adni_1_valid_lists_out:
                print(i)
            # ########################
            time.sleep(3)
            # ########################
            print('### test lists (%d instances):' % len(adni_1_test_lists_out))
            for i in adni_1_test_lists_out:
                print(i)
            # ########################
            time.sleep(3)
            print(len(adni_1_train_lists_out))
            print(len(adni_1_valid_lists_out))
            print(len(adni_1_test_lists_out))
            # ########################
            time.sleep(3)
            # ########################

    return [adni_1_train_lists_out, adni_1_valid_lists_out, adni_1_test_lists_out]


#------------------------------------------------------------------------------------------
# Function: generate Data from lists  
# -> data_params: dict of parameters
# -> selected_label: if you want only generate a specific binary classification
#    example: selected_label="AD_NC" or =None
#------------------------------------------------------------------------------------------

def generate_data_from_lists(data_params, selected_label=None):
    file_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/List_data.pkl'
    data_list = daf.read_lists_from_file(file_path)
    adni_1_in = data_list[0]
    lists_with_names = zip([adni_1_in[0], adni_1_in[1], adni_1_in[2]], ['alz_ADNI_1_train', 'alz_ADNI_1_valid', 'alz_ADNI_1_test'])
    time.sleep(1)
    generate_data_from_selected_dataset(data_params, lists_with_names, selected_label)

#------------------------------------------------------------------------------------------
# Function: generate Data from lists  
#------------------------------------------------------------------------------------------

def generate_data_from_selected_dataset(data_params, lists_with_names, selected_label=None, create_binary_data=True):
    
    print(CP.style.BRIGHT + CP.fg.GREEN + "\n--------------------------------------------------------------------------")
    print(" $ [{} - {} ROI(s)] is selected ... ".format(data_params['3D_or_2D'] , data_params['ROI_list'][data_params['ROI_selection']]))
    print("--------------------------------------------------------------------------\n" + CP.fg.WHITE + CP.style.RESET_ALL)
        
    if create_binary_data:
        queue = []
        if (selected_label is None):  # create All Data
            for (lst, name) in lists_with_names:
                bin_groups = tls.split_lists_to_binary_groups(lst)
                for k in bin_groups:                    
                    label_code = rsd.get_label_binary_codes()[k]
                    queue.append((bin_groups[k], name + '_' + k, label_code))
            for (l, n, c) in queue:
                generate_data(data_params, l, n, c)
        else:
            print("Create lmdbs for : {} ".format(selected_label))
            for (lst, name) in lists_with_names:
                bin_groups = tls.split_lists_to_binary_groups(lst)
                for k in bin_groups:
                    label_code = rsd.get_label_binary_codes()[k]
                    queue.append((bin_groups[k], name + '_' + k, label_code))
            for (l, n, c) in queue:
                for slt in tls.generate_selected_label_list(selected_label):
                    if n == slt:
                        generate_data(data_params, l, n, c)
    else:
        print("create 3 way Data") # extensible for future

#------------------------------------------------------------------------------------------
# generate Data (2D slices patches or 3D Volumes)
#------------------------------------------------------------------------------------------
def generate_data(data_params, lst, data_name, label_code):
    if data_params['3D_or_2D'] == '2D':
        generate_2D_data(data_params, lst, data_name, label_code)
    else:
        generate_3D_data(data_params, lst, data_name, label_code)

#######################################################################################################
# 2D extracting process
#######################################################################################################
def generate_2D_data(data_params, lst, data_name, label_code):
    print(CP.style.BRIGHT + CP.fg.MAGENTA + "--------------------------------------------------------------------------")
    print("> Selected Data: {} for {} - Data size : {}".format(str(data_name).split('_')[3].capitalize(), str(data_name).split('_')[4], len(lst)))
    print("--------------------------------------------------------------------------\n" + CP.fg.WHITE + CP.style.RESET_ALL)    
    process_extracting_2D_data(data_params, lst, data_name, label_code, indice_ROI=data_params['ROI_list'][data_params['ROI_selection']])

#######################################################################################################
# 3D extracting process
#######################################################################################################
def generate_3D_data(data_params, lst, data_name, label_code):
    
    print(CP.style.BRIGHT + CP.fg.MAGENTA + "--------------------------------------------------------------------------")
    print(">  {} Data for [{}] & Data length: [{}]".format(str(data_name).split('_')[3].capitalize(), str(data_name).split('_')[4], len(lst)))
    print("--------------------------------------------------------------------------\n" + CP.fg.WHITE + CP.style.RESET_ALL)    
    process_extracting_3D_data(data_params, lst, data_name, label_code, indice_ROI=data_params['ROI_list'][data_params['ROI_selection']])

#------------------------------------------------------------------------------------------
# 2D extracting 
#------------------------------------------------------------------------------------------
def process_extracting_2D_data(data_params, lst, data_name, label_code, indice_ROI):
    # To do 
    pass
     
#------------------------------------------------------------------------------------------
# 3D extracting 
#------------------------------------------------------------------------------------------
def process_extracting_3D_data(data_params, lst, data_name, label_code, indice_ROI):     
    if("HIPP" in indice_ROI):
        l, r = tls.get_dimensions_cubes_HIPP(data_params) # exctract only Hippocampus ROI
    elif ("PPC" in indice_ROI):
        l, r = tls.get_dimensions_cubes_PPC(data_params) # exctract only Posterior PC ROI
    else:
        # compute both ROIs (in future)
        pass
   
    # get dimensions from the selected ROI (max - min)
    names = ['sag', 'cor', 'axi']
    list_cord_l = [int(l[i+1] - l[i]) for i in range(0, 6, 2)]
    list_cord_r = [int(r[i+1] - r[i]) for i in range(0, 6, 2)]

    # compute the indexes for selected slices
    neighbors = int(data_params['neighbors']) # used for how many of slices we will select  
    sag_l, cor_l, axi_l = [[(int(i/2) - neighbors), (int(i/2)+ neighbors + 1)] for i in { "l_" + str(names[j]) : list_cord_l[j] for j in range(len(list_cord_l))}.values()]
    sag_r, cor_r, axi_r = [[(int(i/2) - neighbors), (int(i/2)+ neighbors + 1)] for i in { "r_" + str(names[j]) : list_cord_r[j] for j in range(len(list_cord_r))}.values()]

    data_selection = str(str(data_name).split('_')[1]).upper() + '_' + str(str(data_name).split('_')[2]).upper()
    data_set = str(data_name).split('_')[3]
    binary_label = str(data_name).split('_')[4]
    target_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/' +indice_ROI + "/3D/"
        
    data_size = 0
    key = 0
    for input_line in lst:        
        # Mean ROI (L & R)
        # data_roi_mean = prc.process_mean_hippocampus(input_line, data_params) # mean cube        
        # cross mean between cubes (in future)       
        # return computed cubes ROIs Left and Right 
        data_roi_left, data_roi_right = prc.process_cube_HIPP(input_line, data_params) # left, right cube
                                           
        # [ID, Date, Class, Age, Sex, MMSE]
        subject_ID = str(input_line[1]).split('/')[7] 
        meta_data = tls.get_meta_data_xml(data_params, subject_ID)
        # print(meta_data, binary_label, data_set, label_code[input_line[0]])

        model_object_normal = HippModel(data_roi_left, data_roi_right, meta_data, int(label_code[input_line[0]]))
        data_size += getsizeof(model_object_normal)
        model_abs_normal_path = target_path + binary_label + '/' + str(data_set) + '/' + str(input_line[0]) + '/' + str(key) + str('_' + indice_ROI + '_').upper() + data_name +'_'+ subject_ID + '_['+ str(input_line[0]) + ']' + str('_normal') + '.pkl'
        # store model data
        daf.save_model(model_object_normal, model_abs_normal_path)

        if  data_params['flip']:
            # Fliped Felt & Right ROI       
            data_roi_left_flip = prc.flip_3d(data_roi_left)
            data_roi_right_flip = prc.flip_3d(data_roi_right)
                        
            #cross fliped model      
            model_object_fliped = HippModel(data_roi_right_flip, data_roi_left_flip, meta_data, int(label_code[input_line[0]]))       
            data_size += getsizeof(model_object_fliped)                    
            model_abs_fliped_path = target_path + binary_label + '/' + str(data_set) + '/' + str(input_line[0]) + '/' + str(key) + str('_' + indice_ROI + '_').upper() + data_name +'_'+ subject_ID + '_['+ str(input_line[0]) + ']' + str('_fliped') + '.pkl'
            # store data model
            daf.save_model(model_object_fliped, model_abs_fliped_path)

        key += 1        
        # Progress of computation       
        print(CP.bg.RED + CP.style.BRIGHT + " {} % percent complete of 100% ".format(round(key/len(lst)*100, 2)) + " " + CP.style.RESET_ALL + CP.bg.RESET, end='\r')
                
    #==========================================================================================================================
    print("\n", end='\r')
    print(CP.style.BRIGHT + "\n>> Data Size is: {} Mb\n".format(data_size/1024) + CP.style.RESET_ALL)  
