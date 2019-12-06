

import pickle
import os
import sys


sys.path.append('/home/karim/workspace/vscode/ADNI_Data_processing/src/data_processing/')



file_name = '/home/karim/workspace/ADNI_workspace/results/ADNI_des/F_28P_F10_MS2_MB10D/HIPP/3D/AD-MCI/test/AD/0_HIPP_alz_ADNI_1_test_AD-MCI_002_S_0619_[AD]_fliped.pkl'

root = '/home/karim/workspace/ADNI_workspace/results/ADNI_des/F_28P_F10_MS2_MB10D/HIPP/3D/MCI-NC/test'



# 1  pickle loader (load one sample)
def pickle_loader(path_file):    
    dir_name = os.path.dirname(path_file)
    with open(path_file, 'rb') as f:
        model_adni = pickle.load(f)
    return model_adni


def find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)




def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images





c, i = find_classes(root)

print(c, i)
images = make_dataset(root, i, extensions='.pkl')

for j in images:
    print(j)

# print(c, i)

# model = pickle_loader(file_name)

# print(model.hippMetaDataVector)