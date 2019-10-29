###########################################################
# ADERGHAL Karim 2019
# Generate lists
##########################################################

from PIL import Image
import scipy.misc
import numpy as np
import services.tools as tls
import io_data.data_acces_file as daf
import interface.inline_print as iprint
import config.config_read as rsd
import config.config as cfg
import services.process as prc
import plot.plot_data as plot_data
import time
import math

#######################################################################################################
#   generate_lists() : function generates and saves list of data of MRI with augmentation
#   (tain, Valid and Test) 
#######################################################################################################
def generate_lists(data_params):
    list_data = []
    file_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/List_data.pkl'
    adni_out = generate_lists_from_adni_dataset(data_params)
    list_data.append(adni_out)
    daf.save_lists_to_file(path_file=file_path, data_list=list_data)








#######################################################################################################
#   split data by using class name (txt file)
#######################################################################################################
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





#######################################################################################################
#   generate lists from ADNI folder dataset
#######################################################################################################
def get_subjects_with_classes(data_params):
    AD, MCI, NC = split_classses_data(daf.read_data_file(str(data_params['adni_1_classes'])))
    iprint.print_adni_desc([AD, MCI, NC])
    time.sleep(1)
    return [AD, MCI, NC]



#######################################################################################################
#   generate lists from ADNI folder dataset
#######################################################################################################
def generate_lists_from_adni_dataset(data_params, augm_test=False, shuffle_data=False, debug=False):  # augm_test bool

    import numpy.random as rnd

    stage_classes = ['AD', 'MCI', 'NC']

    max_blur = float(data_params['sigma'])
    max_shift = int(data_params['shift'])
    default_augm = (0, 0, 0, 0.0)

    adni1_list = get_subjects_with_classes(data_params)
    adni1_size = {'AD': len(adni1_list[0]), 'MCI': len(adni1_list[1]), 'NC': len(adni1_list[2])}
    adni_1_labels = {'AD': adni1_list[0], 'MCI': adni1_list[1], 'NC': adni1_list[2]}
    adni_1_dirs_root = {k: [data_params['adni_1_brain_data'] + '/' + i for i in adni_1_labels[k]] for k in stage_classes}

    # test statement for spliting data (check "SPLIT_SET_PARAMS" in config file)
    if(data_params['static_split']):
        test_selected_size = {k: int(data_params['select_test'][k]) for k in stage_classes}
        valid_selected_size = {k: int(data_params['select_valid'][k]) for k in stage_classes} 
    else:
        test_selected_size = {k: int(data_params['select_test'][k]) for k in stage_classes}
        valid_selected_size = {k: int(math.ceil(int(adni1_size[k]) * 20) / 100.0) for k in stage_classes} 
    

    print('\n\n')
    print('-------------------------------------------------------------------------------------------')
    print('------  source patients ADNI 1 (sMRI)', adni1_size)
    print('-------------------------------------------------------------------------------------------\n')

    train_selected_size = {k: adni1_size[k] - valid_selected_size[k] - test_selected_size[k] for k in stage_classes}    
    # print([train_selected_size[item] for item in train_selected_size.keys()])
    # print([valid_selected_size[item] for item in valid_selected_size.keys()])
    # print([test_selected_size[item] for item in test_selected_size.keys()])

    adni_1_test  = {k: adni_1_dirs_root[k][:int(test_selected_size[k])] for k in stage_classes}
    adni_1_valid = {k: adni_1_dirs_root[k][int(test_selected_size[k]):int(test_selected_size[k]) + int(valid_selected_size[k])] for k in stage_classes}
    adni_1_train = {k: adni_1_dirs_root[k][int(valid_selected_size[k]): int(valid_selected_size[k]) + int(train_selected_size[k])] for k in stage_classes}

    adni_1_train_size_balanced = int(max(train_selected_size.values()) * int(data_params['factor']))
    adni_1_valid_size_balanced = int(max(valid_selected_size.values()) * int(data_params['factor']))
    adni_1_test_size = int(min(test_selected_size.values()))
    
    print('source patients used for train:', train_selected_size)
    print('source patients used for validation:', valid_selected_size)
    print('source patients used for test', test_selected_size)
    print('---------------------------------------------------------------------')
    print('* [train] data will be augmented to  {} samples by each class'.format(adni_1_train_size_balanced))
    print('* [valid] data will be augmented to  {} samples by each class'.format(adni_1_valid_size_balanced))
    print('* [test]  data will be not augmented {} samples by each class'.format(adni_1_test_size))
    print('---------------------------------------------------------------------\n')
    
    
    iprint.print_augmentation_table([adni_1_train_size_balanced, adni_1_valid_size_balanced, adni_1_test_size])
    
  
    adni_1_train_lists_out = []
    adni_1_valid_lists_out = []
    adni_1_test_lists_out = []

    for k in stage_classes:

        adni_1_test_lists = [[k, i + '/MRI/'] for i in adni_1_test[k]]
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
            print len(adni_1_train_lists_out)
            print len(adni_1_valid_lists_out)
            print len(adni_1_test_lists_out)
            # ########################
            time.sleep(3)
            # ########################

    # return []
    return [adni_1_train_lists_out, adni_1_valid_lists_out, adni_1_test_lists_out]



#######################################################################################################
#   generate Data from List 
#######################################################################################################

def generate_lmdb_from_lists(data_params, selected_label=None):
    file_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/List_data.pkl'
    data_list = daf.read_lists_from_file(file_path)
    adni_1_in = data_list[0]
    lists_with_names = zip([adni_1_in[0], adni_1_in[1], adni_1_in[2]], ['alz_ADNI_1_train', 'alz_ADNI_1_valid', 'alz_ADNI_1_test'])
    time.sleep(1)
    generate_lmdb_from_selected_data(data_params, lists_with_names, selected_label)


def generate_lmdb_from_selected_data(data_params, lists_with_names, selected_label=None, create_binary_lmdbs=True):

    if create_binary_lmdbs:
        queue = []
        if (selected_label is None):  # create All lmdbs
            # print("Create all lmdbs")
            for (lst, name) in lists_with_names:
                bin_groups = tls.split_lists_to_binary_groups(lst)
                for k in bin_groups:
                    label_code = rsd.get_label_binary_codes()[k]
                    queue.append((bin_groups[k], name + '_' + k, label_code))
            for (l, n, c) in queue:
                generate_lmdb(data_params, l, n, c)
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
                        generate_lmdb(data_params, l, n, c)

    else:
        print("create 3 way lmdbs")



#######################################################################################################
# genearate Data (2D patches or 3D cubes)
#######################################################################################################
def generate_lmdb(data_params, lst, lmdb_name, label_code):

    print(cfg.bcolors.OKBLUE +"------------------------------------------------------------------------")
    print("--------------- creating of {} data lmdb ... ---------------------------".format(data_params['3D_or_2D']))
    print("------------------------------------------------------------------------\n" + cfg.bcolors.ENDC)
    if data_params['3D_or_2D'] == '2D':
        generate_2D_data(data_params, lst, lmdb_name, label_code)
    else:
        generate_3D_data(data_params, lst, lmdb_name, label_code)





#######################################################################################################
# 2D extracting to do
#######################################################################################################
def generate_2D_data(data_params, lst, lmdb_name, label_code):
    print("--->>  creating data for : \'{}\' - {}, size of list : {} <<--- \n".format(lmdb_name, label_code, len(lst)))
    process_extracting_2D_data(data_params, lst, lmdb_name, label_code, indice_ROI="HIPP")
    print("#================================ End Creating dataset =============================================#\n\n")

#######################################################################################################
# 3D extracting to do
#######################################################################################################
def generate_3D_data(data_params, lst, lmdb_name, label_code):
    print("--->>  creating data for : \'{}\' - {}, size of list : {} <<--- \n".format(lmdb_name, label_code, len(lst)))
    process_extracting_3D_data(data_params, lst, lmdb_name, label_code, indice_ROI="HIPP")
    print("#================================ End Creating dataset =============================================#\n\n")



#######################################################################################################
# 2D extracting for pytorch
#######################################################################################################
def process_extracting_2D_data(data_params, lst, lmdb_name, label_code, indice_ROI="HIPP"):
    

    list_sagittal_data = []
    list_axial_data = []
    list_coronal_data = []
    
    if("HIPP" in indice_ROI):
        l, r = tls.get_dimensions_cubes_HIPP(data_params)
    else:
        l, r = tls.get_dimensions_cubes_PCC(data_params)
        

    l_sag = int(l[1] - l[0])
    l_cor = int(l[3] - l[2])
    l_axi = int(l[5] - l[4])
    r_sag = int(r[1] - r[0])
    r_cor = int(r[3] - r[2])
    r_axi = int(r[5] - r[4])

    slc_index_begin = ((l_sag / 2) - 1)
    slc_index_end = ((l_sag / 2) + 2)

    data_selection = str(str(lmdb_name).split('_')[1]).upper() + '_' + str(str(lmdb_name).split('_')[2]).upper()
    lmdb_set = str(lmdb_name).split('_')[3]
    binary_label = str(lmdb_name).split('_')[4]
    projections_name = ['sagittal', 'coronal', 'axial']
    target_path = data_params['adni_data_des'] + tls.get_convention_name(data_params)
    


    destination_data_sag = target_path + '/' + data_selection + '/' + modality + '/' + str(projections_name[0]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
    label_text_file_sag = destination_data_sag + lmdb_set + '_lmdb.txt'
    destination_data_cor = target_path + '/' + data_selection + '/' + modality + '/' + str(projections_name[1]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
    label_text_file_cor = destination_data_cor + lmdb_set + '_lmdb.txt'
    destination_data_axi = target_path + '/' + data_selection + '/' + modality + '/' + str(projections_name[2]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
    label_text_file_axi = destination_data_axi + lmdb_set + '_lmdb.txt'


    print("#================================ End Creating 3D data  ===========================================#\n\n")
        


     
#######################################################################################################
# 3D extracting for pytorch
#######################################################################################################
def process_extracting_3D_data(data_params, lst, lmdb_name, label_code, indice_ROI="HIPP"): 
    
    if("HIPP" in indice_ROI):
        l, r = tls.get_dimensions_cubes_HIPP(data_params)
    else:
        l, r = tls.get_dimensions_cubes_PPC(data_params)

    print(cfg.bcolors.OKGREEN + "\t-----------------------------------------------------")
    print("\t---------------   {} is selected ...  -------------".format(indice_ROI))
    print("\t-----------------------------------------------------\n" + cfg.bcolors.ENDC)
        
    list_sagittal_data = []
    list_axial_data = []
    list_coronal_data = []
            
    l_sag = int(l[1] - l[0])
    l_cor = int(l[3] - l[2])
    l_axi = int(l[5] - l[4])
    r_sag = int(r[1] - r[0])
    r_cor = int(r[3] - r[2])
    r_axi = int(r[5] - r[4])

    slc_index_begin = ((l_sag / 2) - 1)
    slc_index_end = ((l_sag / 2) + 2)

    data_selection = str(str(lmdb_name).split('_')[1]).upper() + '_' + str(str(lmdb_name).split('_')[2]).upper()
    lmdb_set = str(lmdb_name).split('_')[3]
    binary_label = str(lmdb_name).split('_')[4]
    target_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/' +indice_ROI + "/3D/"
    
    print(target_path + '\n')
    
    
    key = 0
    for input_line in lst:
        data_roi_mean = prc.process_mean_hippocampus(input_line, data_params) # mean cube       
        data_roi_left, data_roi_right = prc.process_cube_HIPP(input_line, data_params) # left, right cube
        
        print("type :", type(data_roi_left))       
        
        print '{}. Hippocampus Roi {} , "{}" , Class : {}'.format(key, data_roi.shape, lmdb_name, label_code[input_line[0]])


        plot_data.plot_HIPP(data_roi_left, 0, slc_index_begin, slc_index_end)
        plot_data.plot_HIPP(data_roi_left, 1, slc_index_begin, slc_index_end)
        plot_data.plot_HIPP(data_roi_left, 2, slc_index_begin, slc_index_end)



    # destination_data_sag = target_path + '/' + data_selection + '/' + modality + '/' + str(projections_name[0]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
    # label_text_file_sag = destination_data_sag + lmdb_set + '_lmdb.txt'
    # destination_data_cor = target_path + '/' + data_selection + '/' + modality + '/' + str(projections_name[1]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
    # label_text_file_cor = destination_data_cor + lmdb_set + '_lmdb.txt'
    # destination_data_axi = target_path + '/' + data_selection + '/' + modality + '/' + str(projections_name[2]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
    # label_text_file_axi = destination_data_axi + lmdb_set + '_lmdb.txt'

        key += 1
        
    print("\n#================================ End Creating 3D data  ===========================================#\n\n")


