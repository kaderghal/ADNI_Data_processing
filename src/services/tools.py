import config.config_read as rsd
import os
import errno
import numpy as np
import numpy.random as rnd
from random import shuffle


"""
# generate a conevntionned name from the config parameters
# example : if factor is 100, dim is 28, max shift is 2 and, gaussian blurr is 1.2
# the name it will be => F_28P_F100_MS02_MB12D
"""
def get_convention_name(data_params):
    dim_pixel = get_dimensions_patch(data_params)[0]
    path_file = 'F_' + str(dim_pixel) + 'P_F' + str(data_params['factor']) + '_MS' + str(data_params['shift']) + '_MB' + str(data_params['sigma']).replace(".", "") + 'D'
    return path_file


# get data for both MRI and DTI


def get_selected_subjects_id(data, k_class_adni, test_size_mri, valid_size_mri, train_size_mri, test_size_md, valid_size_md, train_size_md):
    file_test_text_01 = data['list_id'] + 'db_01/db_01_' + k_class_adni + '.txt'  # only sMRI dataset
    file_test_text_02 = data['list_id'] + 'db_02/db_02_' + k_class_adni + '.txt'  # dataset with MD modality
    list_id_01 = []
    list_id_02 = []
    with open(file_test_text_01, 'r') as fs_01:
        for line in fs_01:
            list_id_01.append(line.rstrip('\n'))

    with open(file_test_text_02, 'r') as fs_02:
        for line in fs_02:
            list_id_02.append(line.rstrip('\n'))

    # rnd.shuffle(list_id_01)
    # rnd.shuffle(list_id_02)

    mri_test = list_id_02[:test_size_mri]
    md_test = list_id_02[:test_size_md]
    current_02 = list_id_02[test_size_md:]
    list_id_01.extend(current_02)
    rnd.shuffle(list_id_01)

    mri_valid = list_id_01[:valid_size_mri]
    mri_train = list_id_01[valid_size_mri:]
    md_valid = current_02[:valid_size_md]
    md_train = current_02[valid_size_md:]

    print(k_class_adni)
    for item in mri_test:
        print(item)

    if len(mri_test) == test_size_mri and len(mri_valid) == valid_size_mri and len(mri_train) == train_size_mri and len(md_test) == test_size_md and len(md_valid) == valid_size_md and len(md_train) == train_size_md:
        return mri_test, mri_valid, mri_train, md_test, md_valid, md_train
    else:
        print('Error : something is wrong ! ')
        return None


#
def get_selected_dirs(data, class_adni, test_size, valid_size, train_size):
    file_test_text_01 = data['list_id'] + 'db_01/db_01_' + class_adni + '.txt'
    file_test_text_02 = data['list_id'] + 'db_02/db_02_' + class_adni + '.txt'
    list_id_01 = []
    list_id_02 = []
    with open(file_test_text_01, 'r') as fs_01:
        for line in fs_01:
            list_id_01.append(line.rstrip('\n'))

    with open(file_test_text_02, 'r') as fs_02:
        for line in fs_02:
            list_id_02.append(line.rstrip('\n'))

    rnd.shuffle(list_id_01)
    rnd.shuffle(list_id_02)
    test = list_id_02[:test_size]
    list_id_01.extend(list_id_02[test_size:])

    rnd.shuffle(list_id_01)
    valid = list_id_01[:valid_size]
    train = list_id_01[valid_size:]

    if len(test) == test_size and len(valid) == valid_size and len(train) == train_size:
        return test, valid, train
    else:
        print('Error : something is wrong ! ')
        return None


def matrix_rotation(mtx):
    import numpy as np
    transpose = np.transpose(np.transpose(mtx))
    rotated = list(reversed(zip(*transpose)))
    return rotated


def create_folder(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def nii_to_array(nii_filename, data_type, fix_nan=True):  # get 3d array from the nii image
    import os
    import nibabel as nib
    import numpy as np
    img = nib.load(nii_filename)
    np_data = img.get_data().astype(data_type)
    if fix_nan:
        np_data = np.nan_to_num(np_data)
    return np_data


def generate_selected_list(label, modal_type):  # ('NC-AD', '')
    lst_data = []
    list_data_set = ['alz_sMRI_train', 'alz_sMRI_valid', 'alz_sMRI_test', 'alz_MD_train', 'alz_MD_valid', 'alz_MD_test']
    for item in list_data_set:
        if modal_type in item:
            str_ = item + '_' + label
            lst_data.append(str_)
    return lst_data


def generate_selected_label_list(label):  # ('NC-AD', '')
    lst_data = []
    list_data_set = ['alz_sMRI_train', 'alz_sMRI_valid', 'alz_sMRI_test', 'alz_MD_train', 'alz_MD_valid', 'alz_MD_test']
    for item in list_data_set:
        str_ = item + '_' + label
        lst_data.append(str_)
    return lst_data


def split_lists_to_binary_groups(lists):
    three_way_labels = list(rsd.get_label_binary_codes()['AD-MCI-NC'].keys())  # {'AD': 0, 'MCI': 1, 'NC': 2}

    bin_labels = {'10': three_way_labels[1] + '-' + three_way_labels[0], '12': three_way_labels[1] + '-' + three_way_labels[2], '20': three_way_labels[2] + '-' + three_way_labels[0]}
    bin_groups = {'10': [], '12': [], '20': []}
    for item in lists:
        if item[0] == three_way_labels[0]:
            bin_groups['10'].append(item)
            bin_groups['20'].append(item)
        if item[0] == three_way_labels[1]:
            bin_groups['10'].append(item)
            bin_groups['12'].append(item)
        if item[0] == three_way_labels[2]:
            bin_groups['12'].append(item)
            bin_groups['20'].append(item)
    return {bin_labels[k]: bin_groups[k] for k in ('10', '12', '20')}


def split_mri_dti(item_list):
    mri_list = [(i[0], i[1], i[3]) for i in item_list]
    dti_list = [(i[0], i[2], i[3]) for i in item_list]
    return mri_list, dti_list


def get_dimensions_patch(data):
    hipp = data['hipp_left']
    padding_size = int(data['padding_size'])
    shift_param = int(data['shift'])
    new_hipp = (
        int(hipp[0]) - 1 - shift_param - padding_size, int(hipp[1]) - 1 - shift_param + padding_size,
        int(hipp[2]) - 1 - shift_param - padding_size, int(hipp[3]) - 1 - shift_param + padding_size,
        int(hipp[4]) - 1 - shift_param - padding_size, int(hipp[5]) - 1 - shift_param + padding_size)
    return (new_hipp[1] - new_hipp[0], new_hipp[3] - new_hipp[2], new_hipp[5] - new_hipp[4])


def generate_augm_params(max_blur, max_shift):
    import numpy.random as rnd
    while True:
        shift_x = rnd.randint(-max_shift, max_shift + 1)
        shift_y = rnd.randint(-max_shift, max_shift + 1)
        shift_z = rnd.randint(-max_shift, max_shift + 1)
        blur_sigma = float(rnd.randint(1000)) / 1000 * max_blur
        if shift_x + shift_y + shift_z + blur_sigma > 0:  # ???
            return shift_x, shift_y, shift_z, blur_sigma



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




##############################################################################################################
# generate parameters (0, 0, 0, 0.0) for each subject
##############################################################################################################
def generate_augm_lists(dirs_with_labels, new_size, max_blur, max_shift, default_augm_params=None):
   
    import numpy.random as rnd
    import math
  
    # pas d'augmentation
    if new_size is None or len(dirs_with_labels) == new_size:  # ?
        return [dwl + [default_augm_params] for dwl in dirs_with_labels]
    
    # for avoid similar augmentation  
    local_param_list = generate_augmentation_parameters_list_v2(max_blur, max_shift)
    print(len(local_param_list))
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

    print("-->", dirs_with_labels[0][0], augm_coeff, len(dirs_with_labels), len(res), i)        
    while i < new_size:
        ridx = rnd.randint(len(dirs_with_labels))
        dwl = dirs_with_labels[ridx]
        res.append(dwl + local_param_list[i])
        i += 1


    for x in res:
        print(x[0], x[3])
    return res





def generate_augm_lists_v2(dirs_with_labels, new_size, max_blur, max_shift, default_augm_params=None):
    import numpy.random as rnd
    import math

    # pas d'augmentation
    if new_size is None or len(dirs_with_labels) == new_size:  # ?
        return [dwl + [default_augm_params] for dwl in dirs_with_labels]

    # print("--------------")
    # print(dirs_with_labels[0][0], len(dirs_with_labels))
    augm_coeff = int(math.floor(new_size / len(dirs_with_labels)))
    # print("coff: ", augm_coeff)

    res = []
    i, j = 0, 0
    for dwl in dirs_with_labels:
        res.append(dwl + [(0, 0, 0, 0.0)])
        # print(i, j,  dwl[1], [(0, 0, 0, 0.0)])
        local_param_list = generate_augmentation_parameters_list_v2(max_blur, max_shift)
        shuffle(local_param_list)
        i += 1
        j += 1
        for _ in range(augm_coeff - 1):
            res.append(dwl + local_param_list.pop())
            # print(i, j, dwl[1], local_param_list.pop())
            i += 1
            j += 1

        j = 0

    local_param_list = generate_augmentation_parameters_list_v2(max_blur, max_shift)
    shuffle(local_param_list)

    for x in range(new_size - len(dirs_with_labels) * augm_coeff):
        ridx = rnd.randint(len(dirs_with_labels))
        dwl = dirs_with_labels[ridx]
        # print(i, "y", j, dwl[1], local_param_list.pop()) 
        res.append(dwl + local_param_list.pop())
        i += 1
        j += 1
         
    return res







def get_dimensions_cubes_HIPP(data):
    full_brain_dimension = [121, 145, 121]  # original dimensions
    crp_l = data['hipp_left']
    crp_r = data['hipp_right']
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


def get_dimensions_cubes_PPC(data):
    full_brain_dimension = [121, 145, 121]  # original dimensions
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







def split_list(liste, number_ele):
    return liste[:number_ele], liste[number_ele:]