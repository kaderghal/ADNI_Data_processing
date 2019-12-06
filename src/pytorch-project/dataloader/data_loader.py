
import numpy as np
import pickle
import os
import sys

import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler
from torchvision.datasets import DatasetFolder

from PIL import Image


ADNI_MODEL_EXTENSIONS = ('.pkl')



# 2 Class Datafolder
class Dataset_ADNI_Folder(DatasetFolder):  
    # Methodes
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None):
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.targets = [s[1] for s in samples]
    
    # __getitem__
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        # sample is objet instance of HippModel (L, R, V, Label)
        return (sample.hippLeft, sample.hippRight, sample.hippMetaDataVector, target) 
   
    # __len__
    def __len__(self):
        return len(self.samples)
        
    # _find_classes
    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]            
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    
# 1  pickle loader (load one sample)
def pickle_loader(path_file):    
    dir_name = os.path.dirname(path_file)
    with open(path_file, 'rb') as f:
        model_adni = pickle.load(f)
    return model_adni

# to check if the file type is allowed 
def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    return has_file_allowed_extension(filename, ADNI_MODEL_EXTENSIONS)

 
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