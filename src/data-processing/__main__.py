#!/usr/bin/python

import sys
import interface.inline_print as iprint
import services.tools as tls
import services.generate_sample_sets as gss
import io_data.data_acces_file as daf
import config.config_read as rsd
import config.config as cfg
import config.ColorPrompt as CP
import time



#------------------------------------------------------------------------------------------
# function::__main__ ::
#------------------------------------------------------------------------------------------
def main():
    print('\n' + CP.bcolors.OKBLUE +'----------------------------------------------------------------------------------------------------------')
    print('---------------------------- Preproccing dataset Alzheimer Diseases ADNI ---------------------------------')
    print('----------------------------------------------------------------------------------------------------------\n' +  CP.bcolors.ENDC)




    # Display Data Parameters
    iprint.print_global_params()
    iprint.print_adni_datasets_path()
    iprint.print_augmentation_params()
    iprint.print_split_params()
    iprint.print_roi_params_hippocampus()
    iprint.print_roi_params_posterior_cc()
    iprint.print_label_binary_codes()
    data_params = rsd.get_all_data_params()
    
    # Computes des Table
    

    HIPP_l, HIPP_r = tls.get_dimensions_cubes_HIPP(data_params)
    PPC_l, PPC_r = tls.get_dimensions_cubes_PPC(data_params)
    iprint.print_dimensions_cubes_HIPP(HIPP_l, HIPP_r )
    iprint.print_dimensions_cubes_PPC(PPC_l, PPC_r)
    
    print('----------------------------------------------------------------------------------------------------------\n\n')
    start_time = time.time()
    localtime = time.localtime(time.time())
    print(CP.bcolors.OKBLUE + "==========================================================================")
    print('=      The dataset will be splitted to Train & Validation & Test         =')
    print('=      Start Time : {}                                  ='.format(time.strftime('%Y-%m-%d %H:%M:%S', localtime)))
    print("==========================================================================" + CP.bcolors.ENDC)
    print("\n")
    
        
        
    exit_input = raw_input('\n' + CP.bcolors.WARNING + 'To change the parameters. exit and update the \"config.py\" file \nto continue press yes (Y/n) ?' + CP.bcolors.ENDC)
    exit_bool = False if str(exit_input).lower() == 'y' else True
    if exit_bool:
        print '\n Exiting ...!  ;) \n'
        sys.exit(1)
    print('\n\n')
    
    #--------------------------------------------------------------------
    # [1] : save parameters from the config file to re-used it
    #         
    #--------------------------------------------------------------------
    daf.save_data_params(data_params)    
    
    #--------------------------------------------------------------------
    # [2] : generate lists 
    #         
    #--------------------------------------------------------------------
    gss.generate_lists(data_params)
    
    #--------------------------------------------------------------------
    # [3] : generate data: 
    #       -> by using the generated lists before [2]   
    #--------------------------------------------------------------------
    gss.generate_data_from_lists(data_params)



#------------------------------------------------------------------------------------------
# Start ->->->->->  
#------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()

