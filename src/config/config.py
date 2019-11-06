#!/usr/bin/python

# ====================================================
# Author: Karim ADERGHAL 
# Year: 2019
# Labs: LaBRI & LabSIV
# for ADNI Dataset : ADNI-1 baseline SMRI 
# ====================================================


#------------------------------------------------------------------------------------------
# Debuging & Time Zone
#------------------------------------------------------------------------------------------
DEBUG = False
TIMEZONE = 'France/Bordeaux'

#------------------------------------------------------------------------------------------
# Author Information 
#------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------
# Root path to local workspace (local Machine)
#------------------------------------------------------------------------------------------
ROOT_PATH_LOCAL_MACHINE = {
    # 'root_machine': '/home/karim/workspace/ADNI_workspace'
    'root_machine': '/home/karim/workspace/ADNI_workspace'

}

#------------------------------------------------------------------------------------------
# Global parameters:
# -> Path to the used Deep learning Framework
# -> Path to the output resutls
#------------------------------------------------------------------------------------------
GLOBAL_PARAMS = {
    'pytorch_root': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/path/to/pythorch/',
    'adni_data_src': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_src/',
    'adni_data_des': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_des/'
}

#------------------------------------------------------------------------------------------
# Dataset Folders : 
# -> brain-data: contains "nii" files (Image for brain)
# -> target-data: contains txt file that gives classes for each subject
# -> meta-data: contains XML files for each subject (used to extract meta-data like : age, sex, mmse etc ...)
#------------------------------------------------------------------------------------------
ADNI_DATASET = {
    'adni_1_brain_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI1/brain-data',
    'adni_1_target_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI1/target-data',
    'adni_1_meta_data': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/ADNI1/meta-data',
}

#------------------------------------------------------------------------------------------
# Classes for Subjects
#------------------------------------------------------------------------------------------
ADNI_CLASSES = {
    'adni_1_classes': 'ADNI_1_classes.txt'
}

#------------------------------------------------------------------------------------------
# Coordinates of the ROIs (Hippocampus region, PPC ... ), selected by using the AAL Atlas.
# -> HIPP: we have two cubes (3D Volume) that contain the Left and Right Hippocampus 
#   !: Note that the default dimension of the cubes are (28*28*28)
#   you can change it by adding padding parameter as follow 28 + (x*2) where x is the pad to add.
#   For example if padding_size (x = 3) then the cubes size become : 34 = 28 + (3*2)
# -> PPC: we will add it soon !!!!
#------------------------------------------------------------------------------------------
ROI_PARAMS_GLOBAL = {
    '3D_or_2D': '3D', # extract data   
    'padding_size': 0,  # =>  28 + (x*2)  
    'neighbors': 1, # number of neighbors of the median slice if 2D is selected
}

ROI_PARAMS_HIPP = {
    'hipp_left': (30, 58, 58, 86, 31, 59),  # min_x,max_x ; min_y,max_y ; min_z,max_z
    'hipp_right': (64, 92, 58, 86, 31, 59),  # calculation model : [coordinates - (index + shift, padding)]
}

ROI_PARAMS_PPC = {
    'ppc_left': (30, 58, 58, 86, 31, 59),  # min_x,max_x ; min_y,max_y ; min_z,max_z
    'ppc_right': (64, 92, 58, 86, 31, 59),  # calculation model : [coordinates - (index + shift, padding)]
}

#------------------------------------------------------------------------------------------
# Augmentation params:
# -> factor F: is a multiplication number to augment data 
# in our case : we have 3 classes so:
# (AD, MCI, NC) ==>> after augmentation ==>> (card(AD) = x, card(MCI) = x, card(NC) = x  // where x = F*max(card(AD), card(MCI), card(NC)))
# -> shift (max shift): its mean that we can generate numbers  a, b, and c in [-s,s] 
# to make translation / (a, b, c)
# -> sigma (max sigma): parameter for gaussian blur
# -> (we can use also rotation, flip to augment data)
#------------------------------------------------------------------------------------------
AUGMENTATION_PARAMS = {
    'shift': 2,  # Max Shift
    'sigma': 1.0,  # Max Sigma for Gaussian Blur
    'factor': 1  # Augmentation Factor 
}

#------------------------------------------------------------------------------------------
# Info to split database to Train, Valid, and Test folders
# -> "static_split" is True  : we use selected number to perform split operation
# -> "static_split" is False : we compute 20% for Valid folder (NB: we keep same number for Test)
#------------------------------------------------------------------------------------------
SPLIT_SET_PARAMS = {
    'static_split': False, # if false we comptue numbers with the %
    'select_valid': {'AD': 30, 'MCI': 72, 'NC': 38},  # %20
    'select_test': {'AD': 40, 'MCI': 40, 'NC': 40} # almost %20
}

#------------------------------------------------------------------------------------------
# Labels Naming system
#------------------------------------------------------------------------------------------
LABELS_CODES = {
    # 2-way classification
    'AD-NC': {'AD': 0, 'NC': 1},
    'AD-MCI': {'AD': 0, 'MCI': 1},
    'MCI-NC': {'MCI': 0, 'NC': 1},
    # 3-way classification
    'AD-MCI-NC': {'AD': 0, 'MCI': 1, 'NC': 2}  
}

