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
    'university': 'Université de Bordeaux (Bordeaux)/ University IBN Zohr (Agadir)',
    'lab': 'LaBRI & LabSIV'
}

#------------------------------------------------------------------------------------------
# Root path to local workspace (local Machine)
#------------------------------------------------------------------------------------------
ROOT_PATH_LOCAL_MACHINE = {
    'root_machine': '/home/karim/workspace/ADNI_workspace'

}



#------------------------------------------------------------------------------------------
# Root path to local workspace (local Machine)
#------------------------------------------------------------------------------------------
ROOT_PATH_LOCAL_MACHINE = {
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


    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 64
    train_number_epochs = 100
    
NETWORK_PARAMS = {
    
    'train_folder' : GLOBAL_PARAMS['adni_data_des'] + '',
    'valid_folder' :, 
    'test_folder'  :
    
}