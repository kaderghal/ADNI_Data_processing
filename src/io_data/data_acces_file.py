
###########################################################
# ADERGHAL Karim 2019
# Data Access Files
##########################################################
import os
# import lmdb # torch
import services.tools as tls



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


"""
# save parameters to re-use it latter in next execution...
"""


def save_data_params(data_params):
    import pickle
    import os
    import errno

    path_file = data_params['adni_data_des'] + tls.get_convention_name(data_params) + '/Data_params.pkl'
    try:
        os.makedirs(os.path.dirname(path_file))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    with open(path_file, 'wb') as f:
        pickle.dump(data_params, f)


def read_data_params(path_file):
    import pickle
    dir_name = os.path.dirname(path_file)
    with open(path_file, 'rb') as f:
        data_params = pickle.load(f)
    return data_params


def read_lists_from_file(path_file):
    import pickle
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


"""
read data from file line by line to a List
"""


def read_data_file(path_file):
    with open(path_file) as f:
        content = f.readlines()

    return [item.strip() for item in content]