#!/usr/bin/python

import config.config_read as rsd
import config.ColorPrompt as CP

#------------------------------------------------------------------------------------------
# Display Data: to print data (Terminal)
#------------------------------------------------------------------------------------------

def print_author_info():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Author Information: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_author_info().items():
        print('\t[' + k + ']: ' + str(v))
    print ("\n")

def print_global_params():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Global parameters: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_global_params().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_adni_datasets_path():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Datasets Images: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_adni_datasets().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_classes_datasets_path():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Classes Datasets Paths: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_classes_datasets().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_augmentation_params():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Augmentation parameters: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_augmentation_params().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_split_params():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Splitting dataset parameters: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_split_params().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_roi_params_global():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Roi Global parameters: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_roi_params_global().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")
        
def print_roi_params_hippocampus():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Roi Hippocampus parameters: " + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_roi_params_hippocampus().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_roi_params_posterior_cc():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Roi Posterior CC parameters :" + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_roi_params_posterior_cc().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")
    
def print_label_binary_codes():
    print(CP.style.BRIGHT + CP.fg.GREEN + "Labels Binary Codes :" + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_label_binary_codes().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_all_params_data():
    print (CP.style.BRIGHT + CP.fg.GREEN + "All parameters Data :" + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in rsd.get_all_data_params().items():
        print('\t[' + k + ']: ' + str(v))
    print("\n")

def print_all_params_data_v2(data):
    print(CP.style.BRIGHT + CP.fg.GREEN + "All parameters Data :" + CP.fg.RESET + CP.style.RESET_ALL)
    for k, v in data.items():
        print('\t {} : {}'.format(k, v))
    print("\n")

def print_dimensions_cubes_HIPP(l, r):
    print(CP.style.BRIGHT + CP.fg.GREEN + "Hippocampus Cube (ROI) dimenssion after the extracting process :" + CP.fg.RESET + CP.style.RESET_ALL)
    print('\tHippocampus L : ({}, {}, {})'.format(l[1] - l[0], l[3] - l[2], l[5] - l[4]))
    print('\tHippocampus R : ({}, {}, {})'.format(r[1] - r[0], r[3] - r[2], r[5] - r[4]))
    print("\n")
        
def print_dimensions_cubes_PPC(l, r):
    print(CP.style.BRIGHT + CP.fg.GREEN + "Posterior CC Cube (ROI) dimenssion after the extracting process :" + CP.fg.RESET + CP.style.RESET_ALL)
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





def print_augmentation_table(data):
    print(CP.style.BRIGHT + CP.fg.RED + "--------------------------------------------------------------------------")
    print("|                        Augmentation description                         ")
    print("--------------------------------------------------------------------------")
    print("|         |        AD         |         MCI         |         NC          ")
    print("--------------------------------------------------------------------------")
    print("|  Train  |    {0} -> ({3})   |    {1} -> ({3})     |     {2} -> ({3})    ".format(data[0][0], data[0][1], data[0][2], data[0][3]))
    print("--------------------------------------------------------------------------")
    print("|  Valid  |    {0} -> ({3})     |    {1} -> ({3})       |     {2} -> ({3})    ".format(data[1][0], data[1][1], data[1][2], data[1][3]))
    print("--------------------------------------------------------------------------")
    print("|  Test   |    {0} -> ({3})     |    {1} -> ({3})       |     {2} -> ({3})    ".format(data[2][0], data[2][1], data[2][2], data[2][3]))
    print("--------------------------------------------------------------------------" + CP.fg.RESET + CP.style.RESET_ALL)

       
def print_datasetDescription(data):
    
    print(CP.style.BRIGHT + CP.fg.CYAN + "----------------------------------------------------------------------------------------------------------")
    print("|                                      ADNI-1 description                                                |")
    print("----------------------------------------------------------------------------------------------------------")
    print("|        #Subject   |   Sex (F/M)       |    Age [min, max]/mean(std)   |    MMSE [min, max]mean/std     |")
    print("----------------------------------------------------------------------------------------------------------")
    print("| AD  |     {}     |     {}         |  {}   |    {}    |".format(data[0][1], data[0][2], data[0][3], data[0][4]))
    print("----------------------------------------------------------------------------------------------------------")
    print("| MCI |     {}     |     {}       |  {}    |    {}    |".format(data[1][1], data[1][2], data[1][3], data[1][4]))
    print("----------------------------------------------------------------------------------------------------------")
    print("| NC  |     {}     |     {}       |  {}   |    {}     |".format(data[2][1], data[2][2], data[2][3], data[2][4]))
    print("----------------------------------------------------------------------------------------------------------\n" + CP.fg.RESET + CP.style.RESET_ALL)
    
    
    
    
    
# def print_2D_or_3D_data():
#     selected_decision = raw_input("Do you want create 3D Data roi or 2D slices ? \n - [0] 3D - [1] 2D \n ")
#     return True if int(selected_decision) == 0 else False
