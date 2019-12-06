

import pickle
import os
import sys

# print(pickle.__doc__)

file_name = '/home/karim/workspace/ADNI_workspace/results/ADNI_des/F_28P_F1_MS2_MB10D/HIPP/3D/AD-MCI/test/AD/0_HIPP_alz_ADNI_1_test_AD-MCI_002_S_0619_[AD]_fliped.pkl'

f = open(file_name, 'rb')

model = pickle.load(f)