###########################################################
# ADERGHAL Karim 2018
# Generate lists
##########################################################

from PIL import Image
import scipy.misc
import numpy as np
import services.tools as tls
import io_data.data_acces_file as daf
import interface.inline_print as iprint
import config.config_read as rsd
import services.process as prc
import time


"""
# generate and save list of the MRI & DTI (tain, Valid and Test) datasets

"""


def generate_lists(data_params):
    list_data = []
    file_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/List_data.pkl'
    adni_out = generate_lists_from_adni_dataset(data_params)
    list_data.append(adni_out)
    daf.save_lists_to_file(path_file=file_path, data_list=list_data)


###

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


def get_subjects_with_classes(data_params):
    a1, b1, c1 = split_classses_data(daf.read_data_file(str(data_params['adni_1_classes'])))
    a2, b2, c2 = split_classses_data(daf.read_data_file(str(data_params['adni_2_classes'])))
    a3, b3, c3 = split_classses_data(daf.read_data_file(str(data_params['adni_3_classes'])))
    iprint.print_adni_desc([a1, b1, c1], [a2, b2, c2], [a3, b3, c3])
    time.sleep(1)
    return [a1, b1, c1], [a2, b2, c2], [a3, b3, c3]


"""


"""


def generate_lists_from_adni_dataset(data_params, augm_test=False, shuffle_data=False, debug=False):  # augm_test bool

    import os
    import numpy.random as rnd

    stage_dirs = {
        'NC': 'NC/',
        'MCI': 'MCI/',
        'AD': 'AD/'
    }

    stage_classes = ['AD', 'MCI', 'NC']

    max_blur = float(data_params['sigma'])
    max_shift = int(data_params['shift'])
    default_augm = (0, 0, 0, 0.0)

    adni1_list, adni2_list, adni3_list = get_subjects_with_classes(data_params)

    adni1_size = {'AD': len(adni1_list[0]), 'MCI': len(adni1_list[1]), 'NC': len(adni1_list[2])}
    adni2_size = {'AD': len(adni2_list[0]), 'MCI': len(adni2_list[1]), 'NC': len(adni2_list[2])}
    adni3_size = {'AD': len(adni3_list[0]), 'MCI': len(adni3_list[1]), 'NC': len(adni3_list[2])}


    adni_23_size = {k: adni2_size[k] + adni3_size[k] for k in stage_classes}

    # print(adni_23_size['AD'], adni_23_size['MCI'], adni_23_size['NC'])

    adni_1_labels = {'AD': adni1_list[0], 'MCI': adni1_list[1], 'NC': adni1_list[2]}
    adni_2_labels = {'AD': adni2_list[0], 'MCI': adni2_list[1], 'NC': adni2_list[2]}
    adni_3_labels = {'AD': adni3_list[0], 'MCI': adni3_list[1], 'NC': adni3_list[2]}


    adni_1_dirs_root = {k: [data_params['adni_1_brain_data'] + '/' + i for i in adni_1_labels[k]] for k in stage_classes}
    adni_2_dirs_root = {k: [data_params['adni_2_brain_data'] + '/' + i for i in adni_2_labels[k]] for k in stage_classes}
    adni_3_dirs_root = {k: [data_params['adni_3_brain_data'] + '/' + i for i in adni_3_labels[k]] for k in stage_classes}

    adni_23_dirs_root = {k: adni_2_dirs_root[k] + adni_3_dirs_root[k] for k in stage_classes}
    

    test_selected_size = {k: int(data_params['test_selected'][k]) for k in stage_classes}
    md_valid_size = {k: int(data_params['md_valid_selected'][k]) for k in stage_classes}
    mri_valid_size = {k: int(data_params['mri_valid_selected'][k]) for k in stage_classes}

    


    print([test_selected_size[item] for item in test_selected_size.keys()])
    print([md_valid_size[item] for item in md_valid_size.keys()])
    print([mri_valid_size[item] for item in mri_valid_size.keys()])




    print '\n\n'
    print '-------------------------------------------------------------------------------------------'
    print '------  source patients ADNI 1 (only sMRI)', adni1_size
    print '-------------------------------------------------------------------------------------------\n'

    valid_adni_1_size = {k: (int(adni1_size[k]) * 20) // 100 for k in stage_classes}
    train_adni_1_size = {k: adni1_size[k] - valid_adni_1_size[k] - test_selected_size[k] for k in stage_classes}

    adni_1_test = {k: adni_1_dirs_root[k][:int(test_selected_size[k])] for k in stage_classes}
    adni_1_valid = {k: adni_1_dirs_root[k][int(test_selected_size[k]):int(test_selected_size[k]) + int(valid_adni_1_size[k])] for k in stage_classes}
    adni_1_train = {k: adni_1_dirs_root[k][int(valid_adni_1_size[k]): int(valid_adni_1_size[k]) + int(train_adni_1_size[k])] for k in stage_classes}

    adni_1_train_size_balanced = int(max(train_adni_1_size.values()) * int(data_params['factor']))
    adni_1_valid_size_balanced = int(max(valid_adni_1_size.values()) * int(data_params['factor']))
    adni_1_test_size = int(min(test_selected_size.values()))

    print('source patients used for train:', train_adni_1_size)
    print('source patients used for validation:', valid_adni_1_size)
    print('source patients used for test', test_selected_size)
    print('---------------------------------------------------------------------')
    print('* [train] data will be augmented to  {} samples by each class'.format(adni_1_train_size_balanced))
    print('* [valid] data will be augmented to  {} samples by each class'.format(adni_1_valid_size_balanced))
    print('* [test]  data will be not augmented {} samples by each class'.format(adni_1_test_size))
    print('---------------------------------------------------------------------\n')

    # adni_1_train_lists_out = []
    # adni_1_valid_lists_out = []
    # adni_1_test_lists_out = []

    # for k in stage_classes:

    #     adni_1_test_lists = [[k, i + '/MRI/'] for i in adni_1_test[k]]
    #     adni_1_valid_lists = [[k, i + '/MRI/'] for i in adni_1_valid[k]]
    #     adni_1_train_lists = [[k, i + '/MRI/'] for i in adni_1_train[k]]

    #     adni_1_test_lists_out += tls.generate_augm_lists(adni_1_test_lists, None, None, None, default_augm_params=default_augm)
    #     adni_1_valid_lists_out += tls.generate_augm_lists(adni_1_valid_lists, adni_1_valid_size_balanced, max_blur, max_shift, default_augm_params=default_augm)
    #     adni_1_train_lists_out += tls.generate_augm_lists(adni_1_train_lists, adni_1_train_size_balanced, max_blur, max_shift, default_augm_params=default_augm)

    #     if shuffle_data:
    #         rnd.shuffle(adni_1_train_lists_out)
    #         rnd.shuffle(adni_1_valid_lists_out)

    #     if debug:
    #         print ('########################### MRI ##########################')
    #         print('### train lists (%d instances):' % len(adni_1_train_lists_out))
    #         for i in adni_1_train_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         # ########################
    #         print('### valid lists (%d instances):' % len(adni_1_valid_lists_out))
    #         for i in adni_1_valid_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         # ########################
    #         print('### test lists (%d instances):' % len(adni_1_test_lists_out))
    #         for i in adni_1_test_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         print len(adni_1_train_lists_out)
    #         print len(adni_1_valid_lists_out)
    #         print len(adni_1_test_lists_out)
    #         # ########################
    #         time.sleep(3)
    #         # ########################

    print '\n\n'
    print '-------------------------------------------------------------------------------------------'
    print '------  source patients ADNI 2&3 ( both Modality)', adni_23_size
    print '-------------------------------------------------------------------------------------------\n'

    valid_adni_23_size = {k: int(md_valid_size[k]) for k in stage_classes}
    train_adni_23_size = {k: adni_23_size[k] - valid_adni_23_size[k] - test_selected_size[k] for k in stage_classes}


    print(test_selected_size)
    print(valid_adni_23_size)
    print(train_adni_23_size)

    adni_23_test = {k: adni_23_dirs_root[k][:int(test_selected_size[k])] for k in stage_classes}
    adni_23_valid = {k: adni_23_dirs_root[k][int(test_selected_size[k]): int(test_selected_size[k]) + int(valid_adni_23_size[k])] for k in stage_classes}
    adni_23_train = {k: adni_23_dirs_root[k][int(test_selected_size[k]) + int(valid_adni_23_size[k]): int(test_selected_size[k]) + int(valid_adni_23_size[k]) + int(train_adni_23_size[k])] for k in stage_classes}


    # print("-------- TEST ----------")
    # for item in adni_23_test.keys():
    #     print "{} \n".format(item)
    #     for i in adni_23_test[item]:
    #         print(i)


    # print("-------- VALID ----------")
    # for item in adni_23_valid.keys():
    #     print "{} \n".format(item)
    #     for i in adni_23_valid[item]:
    #         print(i)

    # print("-------- TRAIN ----------")
    # for item in adni_23_train.keys():
    #     print "{} \n".format(item)
    #     for i in adni_23_train[item]:
    #         print(i)



    adni_23_train_size_balanced = int(max(train_adni_23_size.values()) * int(data_params['factor']))
    adni_23_valid_size_balanced = int(max(valid_adni_23_size.values()) * int(data_params['factor']))
    adni_23_test_size = int(min(test_selected_size.values()))


    print('source patients used for train:', train_adni_23_size)
    print('source patients used for validation:', valid_adni_23_size)
    print('source patients used for test', test_selected_size)

    print('---------------------------------------------------------------------')
    print('* [train] data will be augmented to  {} samples for each class'.format(adni_23_train_size_balanced))
    print('* [valid] data will be augmented to  {} samples for each class'.format(adni_23_valid_size_balanced))
    print('* [test]  data will be not augmented {} samples for each class'.format(adni_23_test_size))
    print('---------------------------------------------------------------------\n')

    # adni_23_train_lists_out = []
    # adni_23_valid_lists_out = []
    # adni_23_test_lists_out = []

    # for k in stage_classes:

    #     print("-------------------------------------------------------")
    #     print("--------------           {}            ----------------".format(k))
    #     print("-------------------------------------------------------")
    #     adni_23_test_lists = [[k, i + '/MRI/', i + '/DTI/'] for i in adni_23_test[k]]
    #     adni_23_valid_lists = [[k, i + '/MRI/', i + '/DTI/'] for i in adni_23_valid[k]]
    #     adni_23_train_lists = [[k, i + '/MRI/', i + '/DTI/'] for i in adni_23_train[k]]



    #     adni_23_test_lists_out += tls.generate_augm_lists(adni_23_test_lists, None, None, None, default_augm_params=default_augm)
    #     adni_23_valid_lists_out += tls.generate_augm_lists(adni_23_valid_lists, adni_23_valid_size_balanced, max_blur, max_shift, default_augm_params=default_augm)
    #     adni_23_train_lists_out += tls.generate_augm_lists(adni_23_train_lists, adni_23_train_size_balanced, max_blur, max_shift, default_augm_params=default_augm)


    #     if shuffle_data:
    #         rnd.shuffle(adni_23_train_lists_out)
    #         rnd.shuffle(adni_23_valid_lists_out)

    #     if debug:
    #         print ('########################### MRI ##########################')
    #         print('### train lists (%d instances):' % len(adni_23_train_lists_out))
    #         for i in adni_23_train_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         # ########################
    #         print('### valid lists (%d instances):' % len(adni_23_valid_lists_out))
    #         for i in adni_23_valid_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         # ########################
    #         print('### test lists (%d instances):' % len(adni_23_test_lists_out))
    #         for i in adni_23_test_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         print len(adni_23_train_lists_out)
    #         print len(adni_23_valid_lists_out)
    #         print len(adni_23_test_lists_out)
    #         # ########################
    #         time.sleep(3)
    #         # #######################



    mri_adni_123_size = {k: int(adni1_size[k]) + int(adni2_size[k]) + int(adni3_size[k]) for k in stage_classes}



    print '\n\n'
    print '-------------------------------------------------------------------------------------------'
    print '------  source patients ADNI 1&2&3 fusion :', mri_adni_123_size
    print '-------------------------------------------------------------------------------------------\n'

    adni_23_test = {k: adni_23_dirs_root[k][:int(test_selected_size[k])] for k in stage_classes}
    rest_23_adni = {k: adni_23_dirs_root[k][int(test_selected_size[k]):] for k in stage_classes}
    adni_1_test = {k: adni_1_dirs_root[k][:int(test_selected_size[k])] for k in stage_classes}
    rest_1_adni = {k: adni_1_dirs_root[k][int(test_selected_size[k]):] for k in stage_classes}
    fusion_adni_123 = {k: rest_23_adni[k] + rest_1_adni[k] for k in stage_classes}

    # ADNI 1 2 3 MRI

    valid_adni_fusion_size = {k: (int(len(fusion_adni_123[k])) * 20) // 100 for k in stage_classes}
    train_adni_fusion_size = {k: int(len(fusion_adni_123[k])) - int(valid_adni_fusion_size[k]) for k in stage_classes}
    adni_1_test_size = {k: len(adni_1_test[k]) for k in stage_classes}

    adni_fusion_valid = {k: fusion_adni_123[k][int(test_selected_size[k]):int(test_selected_size[k]) + int(valid_adni_fusion_size[k])] for k in stage_classes}
    adni_fusion_train = {k: fusion_adni_123[k][int(valid_adni_fusion_size[k]): int(valid_adni_fusion_size[k]) + int(train_adni_fusion_size[k])] for k in stage_classes}

    adni_fusion_train_size_balanced = int(max(train_adni_fusion_size.values()) * int(data_params['factor']))
    adni_fusion_valid_size_balanced = int(max(valid_adni_fusion_size.values()) * int(data_params['factor']))
    adni_fusion_test_size = int(min(adni_1_test_size.values()))

    print('source patients used for train:', train_adni_fusion_size)
    print('source patients used for validation:', valid_adni_fusion_size)
    print('source patients used for test', adni_1_test_size)
    print('---------------------------------------------------------------------')
    print('* [train] data will be augmented to  {} samples by each class'.format(adni_fusion_train_size_balanced))
    print('* [valid] data will be augmented to  {} samples by each class'.format(adni_fusion_valid_size_balanced))
    print('* [test]  data will be not augmented {} samples by each class'.format(adni_fusion_test_size))
    print('---------------------------------------------------------------------\n')

    adni_fusion_train_lists_out = []
    adni_fusion_valid_lists_out = []
    adni_fusion_test_lists_out = []

    # for k in stage_classes:

    #     adni_fusion_test_lists = [[k, i + '/MRI/'] for i in adni_1_test[k]]
    #     adni_fusion_valid_lists = [[k, i + '/MRI/'] for i in adni_fusion_valid[k]]
    #     adni_fusion_train_lists = [[k, i + '/MRI/'] for i in adni_fusion_train[k]]

    #     adni_fusion_test_lists_out += tls.generate_augm_lists(adni_fusion_test_lists, None, None, None, default_augm_params=default_augm)
    #     adni_fusion_valid_lists_out += tls.generate_augm_lists(adni_fusion_valid_lists, adni_fusion_valid_size_balanced, max_blur, max_shift, default_augm_params=default_augm)
    #     adni_fusion_train_lists_out += tls.generate_augm_lists(adni_fusion_train_lists, adni_fusion_train_size_balanced, max_blur, max_shift, default_augm_params=default_augm)

    #     if shuffle_data:
    #         rnd.shuffle(adni_fusion_train_lists_out)
    #         rnd.shuffle(adni_fusion_valid_lists_out)

    #     if debug:
    #         print ('########################### MRI ##########################')
    #         print('### train lists (%d instances):' % len(adni_fusion_train_lists_out))
    #         for i in adni_fusion_train_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         # ########################
    #         print('### valid lists (%d instances):' % len(adni_fusion_valid_lists_out))
    #         for i in adni_fusion_valid_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         # ########################
    #         print('### test lists (%d instances):' % len(adni_fusion_test_lists_out))
    #         for i in adni_fusion_test_lists_out:
    #             print(i)
    #         # ########################
    #         time.sleep(3)
    #         print len(adni_fusion_train_lists_out)
    #         print len(adni_fusion_valid_lists_out)
    #         print len(adni_fusion_test_lists_out)
    #         # ########################
    #         time.sleep(3)
    #         # #######################

    # adni_1_out = [adni_1_train_lists_out, adni_1_valid_lists_out, adni_1_test_lists_out]
    # adni_23_out = [adni_23_train_lists_out, adni_23_valid_lists_out, adni_23_test_lists_out]
    # adni_fusion_out = [adni_fusion_train_lists_out, adni_fusion_valid_lists_out, adni_fusion_test_lists_out]
    # return adni_1_out, adni_23_out, adni_fusion_out
    return [], [], []


"""
generate patches 2D
"""


def generate_lmdb_from_lists(data_params, selected_label=None):
    file_path = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/List_data.pkl'
    data_list = daf.read_lists_from_file(file_path)
    adni_1_in = data_list[0]
    adni_23_in = data_list[1]
    adni_fusion_in = data_list[2]
    lists_with_names = zip([adni_1_in[0], adni_1_in[1], adni_1_in[2], adni_23_in[0], adni_23_in[1], adni_23_in[2], adni_fusion_in[0], adni_fusion_in[1], adni_fusion_in[2]], ['alz_ADNI_1_train', 'alz_ADNI_1_valid', 'alz_ADNI_1_test', 'alz_ADNI_23_train', 'alz_ADNI_23_valid', 'alz_ADNI_23_test', 'alz_ADNI_fusion_train', 'alz_ADNI_fusion_valid', 'alz_ADNI_fusion_test'])
    time.sleep(3)
    generate_lmdb_from_selected_data(data_params, lists_with_names, selected_label)


def generate_lmdb_from_selected_data(data_params, lists_with_names, selected_label=None, create_binary_lmdbs=True):

    if create_binary_lmdbs:
        queue = []
        if (selected_label is None):  # create All lmdbs
            print ("Create all lmdbs")
            for (lst, name) in lists_with_names:
                bin_groups = tls.split_lists_to_binary_groups(lst)
                for k in bin_groups:
                    label_code = rsd.get_label_binary_codes()[k]
                    queue.append((bin_groups[k], name + '_' + k, label_code))
            for (l, n, c) in queue:
                generate_lmdb(data_params, l, n, c)
        else:
            print "Create lmdbs for : {} ".format(selected_label)
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
        print "create 3 way lmdbs"


def generate_lmdb(data_params, lst, lmdb_name, label_code):

    print "------------------------------------------------------------------------"
    print "--------------- creating of {} data lmdb ... ---------------------------".format(data_params['3D_or_2D'])
    print "------------------------------------------------------------------------\n"
    if data_params['3D_or_2D'] == '2D':
        generate_2D_data(data_params, lst, lmdb_name, label_code)
    else:
        generate_3D_data(data_params, lst, lmdb_name, label_code)


"""
3D extracting to do
"""


def generate_3D_data(data_params, lst, lmdb_name, label_code):
    pass


def generate_2D_data(data_params, lst, lmdb_name, label_code):
    print "#========== creating data for : \'{}\' - {}, size of list : {} ================#".format(lmdb_name, label_code, len(lst))

    if str(lmdb_name).split('_')[2] == str(1) or str(lmdb_name).split('_')[2].lower() == str('fusion').lower():
        print "MRI"
        generate_2D_patches(data_params, lst, lmdb_name, label_code, 'MRI', 1)
        # generate_patches_MRI(data_params, lst, lmdb_name, label_code)
    else:
        print "MRI & DTI"
        # generate_patches_MRI_DTI(data_params, lst, lmdb_name, label_code)
        generate_2D_patches(data_params, lst, lmdb_name, label_code, 'DTI', 2)
        generate_2D_patches(data_params, lst, lmdb_name, label_code, 'MRI', 2)

    print "#================================ End Creating dataset =============================================#\n\n"


def generate_2D_patches(data_params, lst, lmdb_name, label_code, modality, indice):

    print "#===================================================================================================#"
    print "#================================ 2D patches Extraction ============================================#"
    print "#===================================================================================================#"

    list_sagittal_data = []
    list_axial_data = []
    list_coronal_data = []
    l, r = tls.get_dimensions_cubes(data_params)
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

    tls.create_folder(destination_data_sag + 'png/')
    tls.create_folder(destination_data_axi + 'png/')
    tls.create_folder(destination_data_cor + 'png/')

    # creating binary folders
    for k, v in label_code.iteritems():
        tls.create_folder(destination_data_sag + k + '_' + str(v) + '/')
        tls.create_folder(destination_data_axi + k + '_' + str(v) + '/')
        tls.create_folder(destination_data_cor + k + '_' + str(v) + '/')

    key = 0
    for input_line in lst:
        if indice == 2:
            if modality == 'MRI':
                input_line = [input_line[0], input_line[1], input_line[3]]
            else:
                input_line = [input_line[0], input_line[2], input_line[3]]

        data_roi = prc.process_mean_hippocampus(input_line, data_params)
        print '{}. Hippocampus Roi {} , "{}" , Class : {}'.format(key, data_roi.shape, lmdb_name, label_code[input_line[0]])

        sub_id_name = str([i for i in str(input_line[1]).split('/') if '_S_' in i][0]) + '_' + str(modality)

        # Sagittal data
        sag_data = data_roi[slc_index_begin:slc_index_end, :, :]  # (3,28,28)
        sag_image = np.transpose(sag_data, (0, 1, 2))
        relative_folder = '/' + data_selection + '/' + modality + '/' + str(projections_name[0]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
        sub_line_1, sub_line_2, sub_line_3 = [str(relative_folder) + str(i) for i in [str(key) + '_0' + str(i) + '_' + lmdb_name + '_' + str(projections_name[0]) + '.png' for i in range(slc_index_begin, ++slc_index_end, 1)]]
        container_png = np.zeros((sag_image.shape[1], sag_image.shape[2], 3))
        container_png[:, :, 0], container_png[:, :, 1], container_png[:, :, 2] = [np.array(tls.matrix_rotation(sag_image[i, :, :])) for i in range(3)]
        destination_image_3k = destination_data_sag + input_line[0] + '_' + str(label_code[input_line[0]]) + '/'
        scipy.misc.imsave(destination_image_3k + str(key) + '_00_3k_' + lmdb_name + '_' + str(projections_name[0]) + '_' + sub_id_name + '.png', container_png)

        # Coronal data
        cor_data = data_roi[:, slc_index_begin:slc_index_end, :]  # (3,28,28)
        cor_image = np.transpose(cor_data, (1, 0, 2))
        relative_folder = '/' + data_selection + '/' + modality + '/' + str(projections_name[1]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
        sub_line_1, sub_line_2, sub_line_3 = [str(relative_folder) + str(i) for i in [str(key) + '_0' + str(i) + '_' + lmdb_name + '_' + str(projections_name[1]) + '.png' for i in range(slc_index_begin, ++slc_index_end, 1)]]
        container_png = np.zeros((cor_image.shape[1], cor_image.shape[2], 3))
        container_png[:, :, 0], container_png[:, :, 1], container_png[:, :, 2] = [np.array(tls.matrix_rotation(cor_image[i, :, :])) for i in range(3)]
        destination_image_3k = destination_data_cor + input_line[0] + '_' + str(label_code[input_line[0]]) + '/'
        scipy.misc.imsave(destination_image_3k + str(key) + '_00_3k_' + lmdb_name + '_' + str(projections_name[1]) + '_' + sub_id_name + '.png', container_png)

        # Axial data
        axi_data = data_roi[:, :, slc_index_begin:slc_index_end]  # (28,28,3) => (3, 28, 28)
        axi_image = np.transpose(axi_data, (2, 0, 1))
        relative_folder = '/' + data_selection + '/' + modality + '/' + str(projections_name[2]) + '/' + binary_label + '/lmdb/' + lmdb_set + '/'
        sub_line_1, sub_line_2, sub_line_3 = [str(relative_folder) + str(i) for i in [str(key) + '_0' + str(i) + '_' + lmdb_name + '_' + str(projections_name[2]) + '.png' for i in range(slc_index_begin, ++slc_index_end, 1)]]
        container_png = np.zeros((axi_image.shape[1], axi_image.shape[2], 3))
        container_png[:, :, 0], container_png[:, :, 1], container_png[:, :, 2] = [np.array(tls.matrix_rotation(axi_image[i, :, :])) for i in range(3)]
        destination_image_3k = destination_data_axi + input_line[0] + '_' + str(label_code[input_line[0]]) + '/'
        scipy.misc.imsave(destination_image_3k + str(key) + '_00_3k_' + lmdb_name + '_' + str(projections_name[2]) + '_' + sub_id_name + '.png', container_png)

        time.sleep(1)
        key += 1

    print "#================================ End Creating 2D patches  ===========================================#\n\n"