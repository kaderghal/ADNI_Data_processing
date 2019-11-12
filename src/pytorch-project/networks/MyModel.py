




import os
import sys
import pickle
import errno
import random
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torchvision import transforms

from torchsummary import summary

import matplotlib.pyplot as plt


# for pickle load
# sys.path.append('/home/karim/workspace/vscode/ADNI_Data_processing/src/data-processing/') # labri Machine
sys.path.append('/home/karim/workspace/vscode-python/ADNI_codesources/kaderghal/src/data-processing/') # home machine


# root_path = '/home/karim/workspace/ADNI_workspace/results/ADNI_des/F_28P_F1_MS2_MB10D/HIPP/3D/AD-NC/' # labri machine
root_path = '/home/karim/workspace/ADNI_workspace/results/ADNI_des/F_28P_F1_MS2_MB10D/HIPP/3D/AD-NC/' # Home machine

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

# function 
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

# Class Datafolder
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
        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        # sample is objet instance from HippModel (L, R, V, Label)
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



# one stream network
class OneStreamNet(nn.Module):
    def __init__(self):
        super(OneStreamNet, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32,  kernel_size=3 ,stride=1, padding=0)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3 ,stride=1, padding=0)

        self.pool1 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2, padding=0)
        self.pool2 = nn.MaxPool3d(kernel_size=(3,3,3), stride=2, padding=0)
        
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        # Defining the fully connected layers
        self.fc1 = nn.Linear(30000, 1024)
        self.fc2 = nn.Linear(1024, 2)
        
    def forward(self, x):
        # x = x.view(32,28,28,28)
        # x = x.view(x.size(0), -1)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)        
        x = self.fc1(x)
        x = self.fc2(x)        
        return x       
        
          

# Network
class Net(nn.Module):
   def __init__(self):
      super().__init__()
      
      self.conv1 = nn.Conv2d(1, 64, 7)
      self.pool1 = nn.MaxPool2d(2)
      self.conv2 = nn.Conv2d(64, 128, 5)
      self.conv3 = nn.Conv2d(128, 256, 5)
      self.linear1 = nn.Linear(2304, 512)      
      self.linear2 = nn.Linear(512, 2)
      
   def forward(self, data):
      res = []
      for i in range(2): # Siamese nets; sharing weights
         x = data[i]
         x = self.conv1(x)
         x = F.relu(x)
         x = self.pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.conv3(x)
         x = F.relu(x)
         
         x = x.view(x.shape[0], -1)
         x = self.linear1(x)
         res.append(F.relu(x))
         
      res = torch.abs(res[1] - res[0])
      res = self.linear2(res)
      return res



# Train function
def train(model, device, train_loader, epoch, optimizer):
    pass
    
    
# Test function
def test(model, device, test_loader):
    pass

#---------------------------------------------------------------
# _________________________   Main   ___________________________
#---------------------------------------------------------------
def main():
    
    # Training params
    num_workers = 1
    num_classes = 2
    save_frequency = 2
    batch_size = 1
    lr = 0.001
    num_epochs = 1
    weight_decay = 0.0001

    # dataset folder loader
    train_data = Dataset_ADNI_Folder(root=root_path + 'train/', loader=pickle_loader, extensions='.pkl', transform=None)
    valid_data = Dataset_ADNI_Folder(root=root_path + 'valid/', loader=pickle_loader, extensions='.pkl', transform=None)
    test_data  = Dataset_ADNI_Folder(root=root_path + 'test/' , loader=pickle_loader, extensions='.pkl', transform=None)

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # select device    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:  {}".format(device))
    
    model = OneStreamNet().to(device)
    summary(model, (1, 28, 28, 28))
    
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []

    # for epoch in range(num_epochs):
    #     # for i, (images, labels) in enumerate(train_loader):
    #     for i, (d1, d2, v, labels) in enumerate(train_loader):
    #         print(i)
            
    #         # Run the forward pass
    #         d1 = torch.unsqueeze(d1, 0).to(device, dtype=torch.float)
    #         outputs = model(d1)
    #         loss = criterion(outputs, labels)
    #         loss_list.append(loss.item())

            # # Backprop and perform Adam optimisation
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # # Track the accuracy
            # total = labels.size(0)
            # _, predicted = torch.max(outputs.data, 1)
            # correct = (predicted == labels).sum().item()
            # acc_list.append(correct / total)

            # if (i + 1) % 100 == 0:
            #     print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
            #         .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
            #             (correct / total) * 100))

    
    
    
    # for epoch in range(num_epochs):
    #      train(model, device, train_loader, epoch, optimizer)
    #      test(model, device, test_loader)
    #      if epoch & save_frequency == 0:
    #         torch.save(model, 'siamese_{:03}.pt'.format(epoch))

    
# __start__ 
if __name__ == '__main__':
   main()