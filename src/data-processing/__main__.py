#!/usr/bin/python

import sys
import interface.inline_print as iprint
import services.tools as tls
import services.generate_sample_sets as gss
import io_data.data_acces_file as daf
import config.config_read as rsd
import config.config_init as cfg
import config.ColorPrompt as CP
import services.process as prc
import time

#------------------------------------------------------------------------------------------
# function::__main__ ::
#------------------------------------------------------------------------------------------
def main():
    print('\n\n' + CP.fg.BLUE +'--------------------------------------------------------------------------')
    print('------------   Preproccing dataset Alzheimer Diseases ADNI   ------------- ')
    print('--------------------------------------------------------------------------\n' +  CP.fg.WHITE )
   
    # Display Data Parameters
    iprint.print_author_info()
    iprint.print_global_params()
    iprint.print_adni_datasets_path()
    iprint.print_augmentation_params()
    iprint.print_split_params()
    iprint.print_roi_params_hippocampus()
    iprint.print_roi_params_posterior_cc()
    iprint.print_label_binary_codes()
    data_params = rsd.get_all_data_params()
    
    # compute dimensions
    HIPP_l, HIPP_r = tls.get_dimensions_cubes_HIPP(data_params)
    PPC_l, PPC_r = tls.get_dimensions_cubes_PPC(data_params)
    iprint.print_dimensions_cubes_HIPP(HIPP_l, HIPP_r )
    iprint.print_dimensions_cubes_PPC(PPC_l, PPC_r)
                   
    #--------------------------------------------------------------------
    # Start execution  Start Timing
    #--------------------------------------------------------------------
    start_time = time.time()
    localtime = time.localtime(time.time())
    print(CP.style.BRIGHT + CP.fg.BLUE + "==========================================================================================================")
    print('=      The dataset will be splitted to Train & Validation & Test         ')
    print('=      Start Time : {}                                  '.format(time.strftime('%Y-%m-%d %H:%M:%S', localtime)))
    print("==========================================================================================================\n" + CP.fg.WHITE + CP.style.RESET_ALL)
    
        
    exit_input = input('\n' + CP.fg.YELLOW + 'To change the parameters. exit and update the \"config.py\" file \nto continue press yes (Y/n) ?' + CP.fg.RESET)
    exit_bool = False if str(exit_input).lower() == 'y' else True
    if exit_bool:
        print(CP.style.BRIGHT + CP.fg.RED + '\n Exiting ...!  ;) \n' + CP.fg.RESET + CP.style.RESET_ALL)
        sys.exit(1)
    print('\n\n')
    
    
    
    #--------------------------------------------------------------------
    # [0] : Computes demoghraphie description Table
    #         
    #--------------------------------------------------------------------
    print(CP.style.BRIGHT + CP.fg.RED + '>$ Computing of Demoghraphy description table. \n' + CP.fg.RESET + CP.style.RESET_ALL)
    time.sleep(3)
    data_desc = prc.compute_demography_description(data_params)
    iprint.print_datasetDescription(data_desc)
    
    
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
    #       -> by using the generated lists in the step before [2]   
    #--------------------------------------------------------------------
    gss.generate_data_from_lists(data_params)

    #--------------------------------------------------------------------
    # Execution finished  
    #--------------------------------------------------------------------
    total_time = round((time.time() - start_time))
    print(CP.style.BRIGHT + CP.fg.BLUE + "==========================================================================================================")
    print('=      Finished Time : {}                               '.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    print('=      Execution Time : {}s / [{}min]                                '.format(total_time, round(total_time/60, 2)))
    print("=========================================================================================================="+ CP.fg.WHITE + CP.style.RESET_ALL)

  
       
       
#------------------------------------------------------------------------------------------
# Start ->>>->>>  
#------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


