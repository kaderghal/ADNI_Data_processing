
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import DatasetFolder
import errno
import numpy as np
import os
import pickle
import random
import sys
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import math




ratio = 3 # reduction ratio for SE 


###############################################################################################################
# server
###############################################################################################################
sys.path.append('/data/ADERGHAL/code-source/ADNI_Data_processing/src/data_processing/')
root_path = '/data/ADERGHAL/ADNI_workspace/results/ADNI_des/F_28P_F10_MS2_MB05D/HIPP/3D/AD-NC/'

###############################################################################################################
# HP computer
###############################################################################################################
#sys.path.append('/home/karim/workspace/vscode-python/ADNI_Data_processing/src/data_processing')
#root_path = '/home/karim/workspace/ADNI_workspace/results/ADNI_des/F_28P_F100_MS2_MB10D/HIPP/3D/AD-NC/'


ADNI_MODEL_EXTENSIONS = ('.pkl')

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
 




#==============================================================================
# Network definition
#==============================================================================

class Network_Baseline(nn.Module):
    def __init__(self):
        super(Network_Baseline, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(28, 32, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=6, stride=1, padding=0),
            nn.ReLU()
            )

        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )


        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU()
            )



        self.fc1 = nn.Linear(128*7*7, 2056)
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(2056, 2)


        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)


        # print("size", x.size())
        x = x.view(-1, self.num_flat_features(x))
        # x = self.dropout(x)
        # print("size", x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features




#==========================================================================
# Function: Main definition 
#========================================================================== 
def main():

    # parames for data
    id_device = 1
    params_num_workers = 4
    batch_size = 64
    num_classes = 2
    save_frequency = 2
    learning_rate = 0.00001
    num_epochs = 500
    weight_decay = 0.0001
    momentum = 0.9
    train_losses, test_losses = [], []
    running_loss = 0
    steps = 0
    print_every = 35 # 175/5
    
    # select device
    device = torch.device("cuda:" + str(id_device) if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    print("using device :", device)
    model = Network_Baseline().to(device)

    # weights initialization    
    # model.apply(weights_init)

    # DataFolder
    train_data = Dataset_ADNI_Folder(root=root_path + 'train/', loader=pickle_loader, extensions='.pkl', transform=None)
    valid_data = Dataset_ADNI_Folder(root=root_path + 'valid/', loader=pickle_loader, extensions='.pkl', transform=None)
    test_data  = Dataset_ADNI_Folder(root=root_path + 'test/' , loader=pickle_loader, extensions='.pkl', transform=None)
    
 
    # Dataloader   
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=params_num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=params_num_workers)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=True, num_workers=params_num_workers)
    valid_loader = test_loader
   
    # net = LeNet()
    # summary(model, (28, 28, 28))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
      
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    valid_acc = []

    running_loss = 0.0
    for epoch in range(num_epochs):
        for i, (d1, d2, v, labels) in enumerate(train_loader):
            
            #
            steps += 1

            # # forward + backward + optimize
            # print("d1 size:", d1.size())
            # d1 = torch.unsqueeze(d1, 1).to(device, dtype=torch.float)
            d1 = d1.to(device, dtype=torch.float)
            # print("d1 size:", d1.size())
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()   

            outputs = model(d1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            # acc_list.append((correct / total) * 100)
            
            
            if steps % print_every == 0:
                acc_list.append((correct / total) * 100)
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for i, (v_d1, v_d2, v_v, v_labels) in enumerate(valid_loader):
                        # v_d1 = torch.unsqueeze(v_d1, 1).to(device, dtype=torch.float)
                        v_d1 = v_d1.to(device, dtype=torch.float)
                        v_labels = v_labels.to(device)
                        v_outputs = model(v_d1)
                        batch_loss = criterion(v_outputs, v_labels)           
                        test_loss += batch_loss.item()                    
                        ps = torch.exp(v_outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            
                # train_losses.append(running_loss/len(train_loader))
                train_losses.append(running_loss/print_every)
                test_losses.append(test_loss/len(valid_loader))    
                
                                
                print(f"Epoch {epoch+1}/{num_epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Train accuracy: {(correct / total) * 100:.3f}.. "
                  f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Test accuracy: {(accuracy/len(valid_loader) * 100):.3f}")
                
                valid_acc.append((accuracy/len(valid_loader) * 100))
                
                running_loss = 0
                model.train()
                
               # scheduler.step()


    
    plt.plot(acc_list, label='Training accu')
    plt.plot(valid_acc, label='Validation accu')
    
    plt.legend(frameon=False)
    plt.show()


    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()



    print('Finished Training')



#==========================================================================
# Start : __Main__
#==========================================================================    
if __name__ == '__main__':
    main()