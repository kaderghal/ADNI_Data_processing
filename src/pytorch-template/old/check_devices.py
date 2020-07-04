#!/usr/bin/python

import math
import torch
import sys
from subprocess import call
import config.ColorPrompt as CP


print(CP.style.BRIGHT + CP.fg.BLUE + "\n=============================================================================================")
print("= Start GPU Information ")
print("=============================================================================================\n" + CP.fg.WHITE + CP.style.RESET_ALL )

print('Python VERSION:', sys.version)
print('pyTorch VERSION:', torch.__version__)
print('CUDA VERSION')
print('CUDNN VERSION:', torch.backends.cudnn.version())
print('# Number CUDA Devices:', torch.cuda.device_count())
print("\n---------------------")
print("| Devices: ")
print("---------------------")
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())


# device index 
id_device = 1

print("\n---------------------")
print("| Selected GPU ")
print("---------------------")
# setting device on GPU if available, else CPU
device = torch.device('cuda:' + str(id_device) if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


#Additional Info when using cuda
if device.type == 'cuda':
    print("Name: ", torch.cuda.get_device_name(id_device))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


print(CP.style.BRIGHT + CP.fg.BLUE + "\n=============================================================================================")
print("= END GPU Information ")
print("=============================================================================================\n" + CP.fg.WHITE + CP.style.RESET_ALL )



