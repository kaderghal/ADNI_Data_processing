###########################################################
# ADERGHAL Karim 2019
# Display data in details
##########################################################

import config.config as cfg
import config.config_read as rsd


def print_author_info():
    print("Author Information :\n")
    for k, v in rsd.get_author_info().iteritems():
        print('\t[' + k + ']: ' + v)
    print ("\n")


def print_global_params():
    print ("Global parameters :")
    for key in rsd.get_global_params():
        print('\t[' + key + ']: ' + rsd.get_global_params()[key])
    print("\n")


def print_adni_datasets_path():
    print("Datasets Images :")
    for k, v in rsd.get_adni_datasets().iteritems():
        print('\t[' + k + ']: ' + v)
    print("\n")


def print_classes_datasets_path():
    print ("Classes Datasets Paths:")
    for key in rsd.get_classes_datasets():
        print('\t[' + key + ']: ' + rsd.get_classes_datasets()[key])
    print("\n")


def print_augmentation_params():
    print ("Augmentation parameters :")
    for key in rsd.get_augmentation_params():
        print('\t[' + key + ']: ' + str(rsd.get_augmentation_params()[key]))
    print("\n")


def print_split_params():
    print("Splitting dataset parameters :")
    for key in rsd.get_split_params():
        print('\t[' + key + ']: ' + str(rsd.get_split_params()[key]))
    print("\n")


def print_roi_params_global():
    print("Roi Global parameters :")
    for key in rsd.get_roi_params_global():
        print('\t[' + key + ']: ' + str(rsd.get_roi_params_global()[key]))
    print("\n")
    
    
    
def print_roi_params_hippocampus():
    print("Roi Hippocampus parameters :")
    for key in rsd.get_roi_params_hippocampus():
        print('\t[' + key + ']: ' + str(rsd.get_roi_params_hippocampus()[key]))
    print("\n")


def print_roi_params_posterior_cc():
    print("Roi Posterior CC parameters :")
    for key in rsd.get_roi_params_posterior_cc():
        print('\t[' + key + ']: ' + str(rsd.get_roi_params_posterior_cc()[key]))
    print("\n")
    
    

def print_label_binary_codes():
    print("Labels Binary Codes :")
    for key in rsd.get_label_binary_codes():
        print('\t[' + key + ']: ' + str(rsd.get_label_binary_codes()[key]))
    print("\n")


def print_all_params_data():
    print ("All parameters Data :")
    for k, v in rsd.get_all_data_params().iteritems():
        print('\t[' + k + ']: ' + str(v))
    print("\n")



def print_all_params_data_v2(data):
    print("All parameters Data :")
    # for k, v in data.iteritems(): print '\t' + k + ' : ' + v
    for k, v in data.iteritems():
        print('\t {} : {}'.format(k, v))
    print("\n")


def print_dimensions_cubes_HIPP(l, r):
    print("Hippocampus Cube (ROI) dimenssion after the extracting process :")
    print('\tHippocampus L : ({}, {}, {})'.format(l[1] - l[0], l[3] - l[2], l[5] - l[4]))
    print('\tHippocampus R : ({}, {}, {})'.format(r[1] - r[0], r[3] - r[2], r[5] - r[4]))
    print("\n")
        
def print_dimensions_cubes_PPC(l, r):
    print("Posterior CC Cube (ROI) dimenssion after the extracting process :")
    print('\tPosterior_CC L : ({}, {}, {})'.format(l[1] - l[0], l[3] - l[2], l[5] - l[4]))
    print('\tPosterior_CC R : ({}, {}, {})'.format(r[1] - r[0], r[3] - r[2], r[5] - r[4]))
    print("\n")   
    
def print_adni_desc(adni1):
    print("\t------------------------------------------------------")
    print("\t|                ADNI Datasets                       |")
    print("\t------------------------------------------------------")
    print("\t----------     AD    |     MCI    |    NC       ------")
    print("\t------------------------------------------------------")
    print("\t| ADNI 1 |     {}   |    {}     |    {}      ------".format(len(adni1[0]), len(adni1[1]), len(adni1[2])))
    print("\t------------------------------------------------------")

def print_2D_or_3D_data():
    selected_decision = raw_input("Do you want create 3D Data roi or 2D slices ? \n - [0] 3D - [1] 2D \n ")
    return True if int(selected_decision) == 0 else False
