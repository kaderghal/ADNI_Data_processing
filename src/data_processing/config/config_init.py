#!/usr/bin/python3

# ====================================================
# Author: Karim ADERGHAL 
# Year: 2019
# Labs: LaBRI & LabSIV
# for ADNI Dataset : ADNI-1 baseline SMRI
# screening selected dataset
# URL: http://adni.loni.usc.edu/ 
# ====================================================


#------------------------------------------------------------------------------------------
# Debuging & Time Zone
#------------------------------------------------------------------------------------------
DEBUG = False
TIMEZONE = 'France/Bordeaux'

#------------------------------------------------------------------------------------------
# Author Informations 
#------------------------------------------------------------------------------------------
AUTHOR_INFO = {
    'author': 'Karim ADERGHAL',
    'name': 'ALZ-ADNI PCS',
    'version': '1.2',
    'year': '2019',
    'description': 'Data Extracting scripts for CNN Alzheimer\'s Disease Classification',
    'url': 'http://github.com/kaderghal',    
    'email': 'aderghal.karim@gmail.com',
    'university': 'University of Bordeaux (Bordeaux)/ University IBN Zohr (Agadir)',
    'lab': 'LaBRI & LabSIV'
}

#------------------------------------------------------------------------------------------
# Root path to local workspace (local Machine)
#------------------------------------------------------------------------------------------
ROOT_PATH_LOCAL_MACHINE = {
    # 'root_machine': '/home/karim/workspace/ADNI_workspace' # HP machine
    'root_machine':'/home/kadergha/ADERGHAL/ADNI_workspace' # Aivcalc4 server

}

#------------------------------------------------------------------------------------------
# Global parameters:
# -> Path to the used Deep learning Framework
# -> Path to the output resutls
#------------------------------------------------------------------------------------------
GLOBAL_PARAMS = {
    'pytorch_root': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/path/to/pythorch/',
    'adni_data_src': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI1_src/',
    'adni_data_des': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI1_des/'
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
    'ROI_selection': 0, # HIPP: 0, PPC: 1, BOTH: 2
    'ROI_list': {0 : 'HIPP', 1 : 'PPC', 2 :'BOTH'},
    '3D_or_2D': '3D', # extract data   
    'padding_size': 0,  # =>  28 + (x*2)  
    'neighbors': 1, # number of neighbors of the median slice if 2D is selected
    'brain_dims': [121, 145, 121] # full brain dimensions (x, y, z)
}

ROI_PARAMS_HIPP = {
    'hipp_left': (30, 58, 58, 86, 31, 59),  # min_x,max_x ; min_y,max_y ; min_z,max_z
    'hipp_right': (64, 92, 58, 86, 31, 59),  # calculation model : [coordinates - (index + shift, padding)]
    # 'hipp_left':  (40, 82 , 82, 124, 40, 82),
    # 'hipp_right': (98, 140, 82, 124, 40, 82),

}

ROI_PARAMS_PPC = { # to calculate from Atlas AAL 
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
    'augm_test': False, #False, #True, # augment Test set
    'shift': 2,  # Max Shift
    'sigma': 0.4,  # Max Sigma for Gaussian Blur
    'factor': 10,  # Augmentation Factor 
    'flip': True #False, #True # excute the flip operation for cubes
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

