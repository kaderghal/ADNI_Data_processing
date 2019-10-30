
import io_data.data_acces_file as daf
import services.tools as tls
import numpy as np


def mean_hipp(mat_a, mat_b):
    x, y, z = mat_a.shape
    for i in range(x):
        for j in range(y):
            for k in range(z):
                mean_hipp[i, j, k] = (mat_a[i, j, k] + mat_b[i, j, k]) / 2
    return mean_hipp


def flip_3d(data):
    x, y, z = data.shape
    hipp_local = np.empty((x, y, z), np.float)
    for i in range(x):
        for j in range(y):
            for k in range(z):
                i_f = ((x - 1) - i)
                hipp_local[i_f, j, k] = data[i, j, k]
    return hipp_local


def mean_3d_matrix(mat_a, mat_b):
    x, y, z = mat_a.shape
    mean_hipp_local = np.empty((x, y, z), np.float)
    for i in range(x):
        for j in range(y):
            for k in range(z):
                mean_hipp_local[i, j, k] = (mat_a[i, j, k] + mat_b[i, j, k]) / 2
    return mean_hipp_local


def crop_cubes(data_l, data_r, crp_l, crp_r):
    cube_hipp_l = data_l[crp_l[0]:crp_l[1], crp_l[2]:crp_l[3], crp_l[4]:crp_l[5]]
    cube_hipp_r = data_r[crp_r[0]:crp_r[1], crp_r[2]:crp_r[3], crp_r[4]:crp_r[5]]
    return cube_hipp_l, cube_hipp_r


def augmentation_cubes(data, max_shift, augm_params):
    # augm_params should be a tuple of 4 elements: shift_x, shift_y, shift_z, blur_sigma
    if data.ndim != 3 or len(augm_params) != 4:
        raise NameError('invalid input')

    from scipy.ndimage.filters import gaussian_filter
    shift_x = augm_params[0]
    shift_y = augm_params[1]
    shift_z = augm_params[2]
    blur_sigma = augm_params[3]
    s_x, s_y, s_z = (data.shape[0] - 2 * max_shift, data.shape[1] - 2 * max_shift, data.shape[2] - 2 * max_shift)
    blurred = data if blur_sigma == 0 else gaussian_filter(data, sigma=blur_sigma)
    sub_data_l = blurred[max_shift + shift_x: s_x + max_shift + shift_x, max_shift + shift_y: s_y + max_shift + shift_y,
                         max_shift + shift_z: s_z + max_shift + shift_z]
    sub_data_r = blurred[max_shift - shift_x: s_x + max_shift - shift_x, max_shift + shift_y: s_y + max_shift + shift_y,
                         max_shift + shift_z: s_z + max_shift + shift_z]
    return sub_data_l, sub_data_r  # return two augmented cubes


def process_mean_hippocampus(list_item, data_params):
    nii = ""
    nii = daf.get_nii_from_folder(list_item[1])[0]  # get first found files (nii) from dir
    array = tls.nii_to_array(nii, np.float)
    padding_param = int(data_params['padding_size'])
    max_shift_param = int(data_params['shift'])

    # Augmentations cubes
    sub_l, sub_r = augmentation_cubes(array, max_shift_param, list_item[2])
    roi_hipp_l_params = data_params['hipp_left']  # Hippocampus ROI corrdinates(x,x,y,y,z,z)
    roi_hipp_r_params = data_params['hipp_right']
    new_crp_l = (roi_hipp_l_params[0] - 1 - max_shift_param - padding_param, roi_hipp_l_params[1] - 1 - max_shift_param + padding_param,
                 roi_hipp_l_params[2] - 1 - max_shift_param - padding_param, roi_hipp_l_params[3] - 1 - max_shift_param + padding_param,
                 roi_hipp_l_params[4] - 1 - max_shift_param - padding_param, roi_hipp_l_params[5] - 1 - max_shift_param + padding_param)
    new_crp_r = (roi_hipp_r_params[0] - 1 - max_shift_param - padding_param, roi_hipp_r_params[1] - 1 - max_shift_param + padding_param,
                 roi_hipp_r_params[2] - 1 - max_shift_param - padding_param, roi_hipp_r_params[3] - 1 - max_shift_param + padding_param,
                 roi_hipp_r_params[4] - 1 - max_shift_param - padding_param, roi_hipp_r_params[5] - 1 - max_shift_param + padding_param)
    # print 'crop params L : ({}, {}, {}, {}, {}, {})'.format(new_crp_l[0], new_crp_l[1], new_crp_l[2], new_crp_l[3], new_crp_l[4], new_crp_l[5])
    # print 'crop params R : ({}, {}, {}, {}, {}, {})'.format(new_crp_r[0], new_crp_r[1], new_crp_r[2], new_crp_r[3], new_crp_r[4], new_crp_r[5])
    roi_cube_left, roi_cube_right = crop_cubes(sub_l, sub_r, new_crp_l, new_crp_r)
    roi_cube_right_flipped = flip_3d(roi_cube_right)
    roi_hippocampus_mean = mean_3d_matrix(roi_cube_left, roi_cube_right_flipped)
    return roi_hippocampus_mean




def process_cube_HIPP(list_item, data_params):
    nii = ""
    nii = daf.get_nii_from_folder(list_item[1])[0]  # get first found file (nii) from dir
    array = tls.nii_to_array(nii, np.float)
    padding_param = int(data_params['padding_size'])
    max_shift_param = int(data_params['shift'])

    # Augmentations cubes
    sub_l, sub_r = augmentation_cubes(array, max_shift_param, list_item[2])
    roi_hipp_l_params = data_params['hipp_left']  # Hippocampus ROI corrdinates(x,x,y,y,z,z)
    roi_hipp_r_params = data_params['hipp_right']
    new_crp_l = (roi_hipp_l_params[0] - 1 - max_shift_param - padding_param, roi_hipp_l_params[1] - 1 - max_shift_param + padding_param,
                 roi_hipp_l_params[2] - 1 - max_shift_param - padding_param, roi_hipp_l_params[3] - 1 - max_shift_param + padding_param,
                 roi_hipp_l_params[4] - 1 - max_shift_param - padding_param, roi_hipp_l_params[5] - 1 - max_shift_param + padding_param)
    new_crp_r = (roi_hipp_r_params[0] - 1 - max_shift_param - padding_param, roi_hipp_r_params[1] - 1 - max_shift_param + padding_param,
                 roi_hipp_r_params[2] - 1 - max_shift_param - padding_param, roi_hipp_r_params[3] - 1 - max_shift_param + padding_param,
                 roi_hipp_r_params[4] - 1 - max_shift_param - padding_param, roi_hipp_r_params[5] - 1 - max_shift_param + padding_param)

    return crop_cubes(sub_l, sub_r, new_crp_l, new_crp_r)
















def process_cube_PPC(list_item, data_params):
    nii = ""
    nii = daf.get_nii_from_folder(list_item[1])[0]  # get first found files (nii) from dir
    array = tls.nii_to_array(nii, np.float)
    padding_param = int(data_params['padding_size'])
    max_shift_param = int(data_params['shift'])

    # Augmentations cubes
    sub_l, sub_r = augmentation_cubes(array, max_shift_param, list_item[2])
    roi_ppc_l_params = data_params['ppc_left']  # PCC ROI corrdinates(x,x,y,y,z,z)
    roi_pcc_r_params = data_params['ppc_right']
    new_crp_l = (roi_ppc_l_params[0] - 1 - max_shift_param - padding_param, roi_ppc_l_params[1] - 1 - max_shift_param + padding_param,
                 roi_ppc_l_params[2] - 1 - max_shift_param - padding_param, roi_ppc_l_params[3] - 1 - max_shift_param + padding_param,
                 roi_ppc_l_params[4] - 1 - max_shift_param - padding_param, roi_ppc_l_params[5] - 1 - max_shift_param + padding_param)
    new_crp_r = (roi_pcc_r_params[0] - 1 - max_shift_param - padding_param, roi_pcc_r_params[1] - 1 - max_shift_param + padding_param,
                 roi_pcc_r_params[2] - 1 - max_shift_param - padding_param, roi_pcc_r_params[3] - 1 - max_shift_param + padding_param,
                 roi_pcc_r_params[4] - 1 - max_shift_param - padding_param, roi_pcc_r_params[5] - 1 - max_shift_param + padding_param)

    return crop_cubes(sub_l, sub_r, new_crp_l, new_crp_r)