import module_config as cfg

def get_global_params():
    tempo_dict = {}
    tempo_dict['pytorch_root'] = str(cfg.GLOBAL_PARAMS['pytorch_root'])
    tempo_dict['adni_data_src'] = str(cfg.GLOBAL_PARAMS['adni_data_src'])
    tempo_dict['adni_data_des'] = str(cfg.GLOBAL_PARAMS['adni_data_des'])
    return tempo_dict
