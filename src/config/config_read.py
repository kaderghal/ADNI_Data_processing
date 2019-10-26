import config as cfg


"""
# get_author_info
"""


def get_author_info():
    tempo_dict = {}
    tempo_dict['name'] = str(cfg.AUTHOR_INFO['name'])
    tempo_dict['version'] = str(cfg.AUTHOR_INFO['version'])
    tempo_dict['year'] = str(cfg.AUTHOR_INFO['year'])
    tempo_dict['description'] = str(cfg.AUTHOR_INFO['description'])
    tempo_dict['url'] = str(cfg.AUTHOR_INFO['url'])
    tempo_dict['author'] = str(cfg.AUTHOR_INFO['author'])
    tempo_dict['email'] = str(cfg.AUTHOR_INFO['email'])
    tempo_dict['lab'] = str(cfg.AUTHOR_INFO['lab'])
    return tempo_dict


def get_global_params():
    tempo_dict = {}
    tempo_dict['pytorch_root'] = str(cfg.GLOBAL_PARAMS['pytorch_root'])
    tempo_dict['adni_data_src'] = str(cfg.GLOBAL_PARAMS['adni_data_src'])
    tempo_dict['adni_data_des'] = str(cfg.GLOBAL_PARAMS['adni_data_des'])
    return tempo_dict


def get_adni_datasets():
    tempo_dict = {}
    tempo_dict['adni_1_brain_data'] = str(cfg.ADNI_DATASET['adni_1_brain_data'])
    tempo_dict['adni_1_target_data'] = str(cfg.ADNI_DATASET['adni_1_target_data'])
    return tempo_dict


def get_classes_datasets():
    tempo_dict = {}
    tempo_dict['adni_1_classes'] = str(cfg.ADNI_DATASET['adni_1_target_data']) + '/' + str(cfg.ADNI_CLASSES['adni_1_classes'])
    return tempo_dict


def get_roi_params_hippocampus():
    tempo_dict = {}
    tempo_dict['3D_or_2D'] = cfg.ROI_PARAMS_HIPP['3D_or_2D']
    tempo_dict['hipp_left'] = cfg.ROI_PARAMS_HIPP['hipp_left']
    tempo_dict['hipp_right'] = cfg.ROI_PARAMS_HIPP['hipp_right']
    tempo_dict['padding_size'] = int(cfg.ROI_PARAMS_HIPP['padding_size'])
    return tempo_dict

def get_roi_params_posterior_cc():
    tempo_dict = {}
    tempo_dict['3D_or_2D'] = cfg.ROI_PARAMS_PPC['3D_or_2D']
    tempo_dict['ppc_left'] = cfg.ROI_PARAMS_PPC['ppc_left']
    tempo_dict['ppc_right'] = cfg.ROI_PARAMS_PPC['ppc_right']
    tempo_dict['padding_size'] = int(cfg.ROI_PARAMS_PPC['padding_size'])
    return tempo_dict


def get_augmentation_params():
    tempo_dict = {}
    tempo_dict['shift'] = cfg.AUGMENTATION_PARAMS['shift']
    tempo_dict['sigma'] = cfg.AUGMENTATION_PARAMS['sigma']
    tempo_dict['factor'] = cfg.AUGMENTATION_PARAMS['factor']
    return tempo_dict


def get_split_params():
    tempo_dict = {}
    tempo_dict['static_split'] = cfg.SPLIT_SET_PARAMS['static_split']
    tempo_dict['select_valid'] = cfg.SPLIT_SET_PARAMS['select_valid']
    tempo_dict['select_test'] = cfg.SPLIT_SET_PARAMS['select_test']
    return tempo_dict
       
    

def get_label_binary_codes():
    tempo_dict = {}
    tempo_dict['AD-NC'] = cfg.LABELS_CODES['AD-NC']
    tempo_dict['AD-MCI'] = cfg.LABELS_CODES['AD-MCI']
    tempo_dict['MCI-NC'] = cfg.LABELS_CODES['MCI-NC']
    tempo_dict['AD-MCI-NC'] = cfg.LABELS_CODES['AD-MCI-NC']
    return tempo_dict


def get_all_data_params():
    lst_all = {}

    for item in get_global_params():
        lst_all[item] = get_global_params()[item]

    for item in get_adni_datasets():
        lst_all[item] = get_adni_datasets()[item]

    for item in get_classes_datasets():
        lst_all[item] = get_classes_datasets()[item]

    for item in get_roi_params_hippocampus():
        lst_all[item] = get_roi_params_hippocampus()[item]
        
    for item in get_roi_params_posterior_cc():
        lst_all[item] = get_roi_params_posterior_cc()[item]

    for item in get_augmentation_params():
        lst_all[item] = get_augmentation_params()[item]

    for item in get_label_binary_codes():
        lst_all[item] = get_label_binary_codes()[item]

    for item in get_split_params():
        lst_all[item] = get_split_params()[item]

    return lst_all