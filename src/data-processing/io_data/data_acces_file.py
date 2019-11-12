#!/usr/bin/python

import os
import services.tools as tls
import pickle
import errno
# import lmdb # torch

#------------------------------------------------------------------------------------------
# DAF: Data access File: Files & Folders processsing
#------------------------------------------------------------------------------------------

def get_nii_from_folder(folder):
    res = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('stretched.nii'): # or file.endswith('MD.nii'):
                res.append(os.path.join(root, file))
    if len(res) > 1:
        print('WARNING. Folder %s contains more than one files' % folder)
    return res

def initiate_lmdb(folder_path, lmdb_name, drop_existing=False):  # save data to lmdb Folder
    saving_path = folder_path + '/' + lmdb_name
    print("saving_path : ", saving_path)
    if drop_existing:
        import os
        import shutil
        if os.path.exists(lmdb_name):
            shutil.rmtree(lmdb_name)
    env = lmdb.open(lmdb_name, map_size=int(1e12))
    # print('database debug info:', env.stat())
    return env

#------------------------------------------------------------------------------------------
# Save parameters to file to use it later in call 
#------------------------------------------------------------------------------------------

def save_data_params(data_params):
    path_file = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/Data_params.pkl'
    try:
        os.makedirs(os.path.dirname(path_file))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path_file, 'wb') as f:
        pickle.dump(data_params, f)

#------------------------------------------------------------------------------------------
# Read parameters from the file "Data_params.pkl"  
#------------------------------------------------------------------------------------------

def read_data_params(path_file):
    import pickle
    dir_name = os.path.dirname(path_file)
    with open(path_file, 'rb') as f:
        data_params = pickle.load(f)
    return data_params


def read_lists_from_file(path_file):
    dir_name = os.path.dirname(path_file)
    with open(path_file, 'rb') as f:
        data_list = pickle.load(f)
    return data_list


def save_lists_to_file(path_file, data_list):
    import pickle
    import os
    import errno
    try:
        os.makedirs(os.path.dirname(path_file))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path_file, 'wb') as f:
        pickle.dump(data_list, f)

#------------------------------------------------------------------------------------------
# read data from file line by line to a List
#------------------------------------------------------------------------------------------

def read_data_file(path_file):
    with open(path_file) as f:
        content = f.readlines()
    return [item.strip() for item in content]

#------------------------------------------------------------------------------------------
# Save Model to Local Machine 
#------------------------------------------------------------------------------------------

def save_model(model, path_file):
    try:
        os.makedirs(os.path.dirname(path_file))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path_file, 'wb') as f:
        pickle.dump(model, f)

#------------------------------------------------------------------------------------------
# Read Model to Local Machine 
#------------------------------------------------------------------------------------------

def read_model(path_file):
    dir_name = os.path.dirname(path_file)
    with open(path_file, 'rb') as f:
        model = pickle.load(f)
    return model

#------------------------------------------------------------------------------------------
# Save Desciption Demography outpu data to txt file
#------------------------------------------------------------------------------------------
def save_desc_table(data_params, text_data):
    classes = ['AD ', 'MCI', 'NC ']         
    path_file = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/Desciption_ADNI_demography.txt'
    try:
        os.makedirs(os.path.dirname(path_file))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path_file, 'w') as f:
        f.write("----------------------------------------------------------------------------------------------------------\n")
        f.write("|                                      ADNI-1 description                                                |\n")
        f.write("----------------------------------------------------------------------------------------------------------\n")
        f.write("|        #Subject   |   Sex (F/M)       |    Age [min, max]/mean(std)   |    MMSE [min, max]mean/std     |\n")
        f.write("----------------------------------------------------------------------------------------------------------\n")
        for i in range(3):
            f.write("|  {}  |     {}     |     {}         |  {}   |    {}    |\n".format(classes[i], text_data[i][1], text_data[i][2], text_data[i][3], text_data[i][4]))      
        f.write("----------------------------------------------------------------------------------------------------------\n")
    f.close()