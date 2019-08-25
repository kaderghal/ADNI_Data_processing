# ====================================================
# Copyright Karim ADERGHAL 2019
# MIT License
# ====================================================

""" Configuration values needed to parametrize the application """


DEBUG = False
TIMEZONE = 'Morocco/Agadir'


AUTHOR_INFO = {
    'name': 'Alz_ADNI_process',
    'version': '1.3',
    'year': '2018',
    'description': 'Extracting data for CNN Alzheimer\'s Disease Classification',
    'url': 'http://github.com/sabako123/---,
    'author': 'Karim ADERGHAL',
    'email': 'aderghal.karim@gmail.com',
    'lab': 'LaBRI'
}


"""
# Root path to local workspace (local Machine)
"""
ROOT_PATH_LOCAL_MACHINE = {
    'root_machine': '/home/kadergha/ADERGHAL/Datasets/02_ADNI_Datasets/ADNI'

}

"""
# Global parameters for CAFFE FrameWorke Installation folder
# Data Folder src and des
"""
GLOBAL_PARAMS = {
    'root_caffe': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/path/to/caffe/',
    'adni_data_src': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_src/',
    'adni_data_des': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_des/'
}

ADNI_DATASETS = {
    'adni_1_brain_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI1/brain-data',
    'adni_1_target_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI1/target-data',

    'adni_2_brain_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI2/brain-data',
    'adni_2_target_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI2/target-data',

    'adni_3_brain_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI3/brain-data',
    'adni_3_target_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI3/target-data'

}

ADNI_CLASSES = {
    'adni_1_classes': 'ADNI_1_classes.txt',
    'adni_2_classes': 'ADNI_2_classes.txt',
    'adni_3_classes': 'ADNI_3_classes.txt'
}


"""
# coordinates of the ROI (Hippocampus region), selected by using the AAL Atlas.
# two cubes that contain the hippocampus Left and Right
# !: Note that the default dimension of the cubes are (28*28*28)
# you can change it by adding padding parameter as follow 28 + (x*2) with x the pad.
# For example if padding_size = 3 then the cubes size will be : 34 = 28 + (3*2)
"""
ROI_PARAMS_HIPP = {
    '3D_or_2D': '2D', # extract 
    'hipp_left': (30, 58, 58, 86, 31, 59),  # min_x,max_x ; min_y,max_y ; min_z,max_z
    'hipp_right': (64, 92, 58, 86, 31, 59),  # calculation model : [coordinates - (index + shift, padding)]
    'padding_size': 0,  # =>  28 + (x*2)
}

ROI_PARAMS_PPC = {
    '3D_or_2D': '2D', # extract 
    'hipp_left': (30, 58, 58, 86, 31, 59),  # min_x,max_x ; min_y,max_y ; min_z,max_z
    'hipp_right': (64, 92, 58, 86, 31, 59),  # calculation model : [coordinates - (index + shift, padding)]
    'padding_size': 0,  # =>  28 + (x*2)
}

"""
# Augmentation params
# max shift for each projection (s,s,s) with y an integer in [-s,s]
# and max siga parameter for gaussian blur
"""
AUGMENTATION_PARAMS = {
    'shift': 2,  # Max Shift
    'sigma': 1.2,  # Max Sigma for Gaussian Blur
    'factor': 100  # Augmentation Factor 
}

"""
# default split train, validation & test sets
# paameters for splitting the dataset for train, validation and test sets
"""
SPLIT_SET_PARAMS = {
    'mri_valid_selected': {'AD': 37, 'MCI': 79, 'NC': 45},  # %20
    'md_valid_selected': {'AD': 10, 'MCI': 10, 'NC': 10},  # (AD, MCI, NC) selected number for each class
    'test_selected': {'AD': 20, 'MCI': 20, 'NC': 20}
}


"""
# get binary labels code
"""
LABELS_CODES = {
    # 2-way classification
    'AD-NC': {'AD': 0, 'NC': 1},
    'AD-MCI': {'AD': 0, 'MCI': 1},
    'MCI-NC': {'MCI': 0, 'NC': 1},
    # 3-way classification
    'AD-MCI-NC': {'AD': 0, 'MCI': 1, 'NC': 2}  
}

