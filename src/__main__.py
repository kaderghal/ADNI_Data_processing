#!/usr/bin/

import sys
import interface.inline_print as iprint
import services.tools as tls
# import services.generate_sample_sets as gss
import io_data.data_acces_file as daf
import config.config_read as rsd
import time




if __name__ == '__main__':

    print('----------------------------------------------------------------------------------------------------------')
    print('---------------------------- Preproccing dataset Alzheimer Diseases ADNI ---------------------------------')
    print('----------------------------------------------------------------------------------------------------------')


    iprint.print_global_params()
    iprint.print_adni_datasets_path()
    iprint.print_augmentation_params()
    iprint.print_split_params()
    iprint.print_roi_params_hippocampus()
    iprint.print_roi_params_posterior_cc()
    iprint.print_label_binary_codes()
    data_params = rsd.get_all_data_params()
    
    
    
    HIPP_l, HIPP_r = tls.get_dimensions_cubes_HIPP(data_params)
    PCC_l, PCC_r = tls.get_dimensions_cubes_PCC(data_params)
    iprint.print_dimensions_cubes_HIPP(HIPP_l, HIPP_r )
    iprint.print_dimensions_cubes_PCC(PCC_l, PCC_r)
    
    
    

    print('----------------------------------------------------------------------------------------------------------\n\n')
    start_time = time.time()
    localtime = time.localtime(time.time())
    print("==========================================================================")
    print('=      The dataset will be splitted to Train & Validation & Test         =')
    print('=      Start Time : {}                                  ='.format(time.strftime('%Y-%m-%d %H:%M:%S', localtime)))
    print("==========================================================================")
    print("\n")
    
    
    
    
    # exit_input = raw_input('\nTo change the parameters. exit and update the \"config.py\" file \nto continue press yes (Y/n) ?')
    # exit_bool = False if str(exit_input).lower() == 'y' else True
    # if exit_bool:
    #     print '\n Exiting ...!  ;) \n'
    #     sys.exit(1)
    # print '\n\n'
    # 1
    # daf.save_data_params(data_params)
    
    # gss.generate_lists(data_params)