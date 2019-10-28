#!/usr/bin/python3
# ====================================================
# Copyright Karim ADERGHAL 2019
# SMRI dataset
# ====================================================

""" Configuration values needed to parametrize the application """



DEBUG = False
TIMEZONE = 'France/Bordeaux'


AUTHOR_INFO = {
    'name': 'Alz_ADNI_process',
    'version': '1.3',
    'year': '2019',
    'description': 'Extracting data for CNN Alzheimer\'s Disease Classification',
    'url': 'http://github.com/kaderghal',
    'author': 'Karim ADERGHAL',
    'email': 'aderghal.karim@gmail.com',
    'university': 'Bordeaux',
    'lab': 'LaBRI'
}


"""
# Root path to local workspace (local Machine)
"""
ROOT_PATH_LOCAL_MACHINE = {
    'root_machine': '/home/karim/workspace/ADNI_workspace'

}

"""
# Global parameters for pytorch FrameWorke Installation folder
# Data Folder src and des
"""
GLOBAL_PARAMS = {
    'pytorch_root': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/path/to/pythorch/',
    'adni_data_src': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_src/',
    'adni_data_des': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_des/'
}

ADNI_DATASET = {
    'adni_1_brain_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI1/brain-data',
    'adni_1_target_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI1/target-data',
}

ADNI_CLASSES = {
    'adni_1_classes': 'ADNI_1_classes.txt'
}


"""
# coordinates of the ROI (Hippocampus region), selected by using the AAL Atlas.
# two cubes that contain the hippocampus Left and Right
# !: Note that the default dimension of the cubes are (28*28*28)
# you can change it by adding padding parameter as follow 28 + (x*2) with x the pad.
# For example if padding_size = 3 then the cubes size will be : 34 = 28 + (3*2)
"""
ROI_PARAMS_GLOBAL = {
    '3D_or_2D': '3D', # extract data   
    'padding_size': 0,  # =>  28 + (x*2)  
}

ROI_PARAMS_HIPP = {
    'hipp_left': (30, 58, 58, 86, 31, 59),  # min_x,max_x ; min_y,max_y ; min_z,max_z
    'hipp_right': (64, 92, 58, 86, 31, 59),  # calculation model : [coordinates - (index + shift, padding)]
}

ROI_PARAMS_PPC = {
    'ppc_left': (30, 58, 58, 86, 31, 59),  # min_x,max_x ; min_y,max_y ; min_z,max_z
    'ppc_right': (64, 92, 58, 86, 31, 59),  # calculation model : [coordinates - (index + shift, padding)]
}

"""
# Augmentation params
# max shift for each projection (s,s,s) with y an integer in [-s,s]
# and max siga parameter for gaussian blur
"""
AUGMENTATION_PARAMS = {
    'shift': 2,  # Max Shift
    'sigma': 1.2,  # Max Sigma for Gaussian Blur
    'factor': 1  # Augmentation Factor 
}

"""
# default split train, validation & test sets
# paameters for splitting the dataset for train, validation and test sets
"""
SPLIT_SET_PARAMS = {
    'static_split': False, # if false we comptue numbers with the %
    'select_valid': {'AD': 30, 'MCI': 72, 'NC': 38},  # %20
    'select_test': {'AD': 40, 'MCI': 40, 'NC': 40} # almost %20

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

# for text colot
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'