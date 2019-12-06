

#------------------------------------------------------------------------------------------
# Root path to local workspace (local Machine)
#------------------------------------------------------------------------------------------
ROOT_PATH_LOCAL_MACHINE = {
    'root_machine': '/home/karim/workspace/ADNI_workspace'

}

#------------------------------------------------------------------------------------------
# Global parameters:
# -> Path to the used Deep learning Framework
# -> Path to the output resutls
#------------------------------------------------------------------------------------------
GLOBAL_PARAMS = {
    'pytorch_root': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/path/to/pythorch/',
    'adni_data_src': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_src/',
    'adni_data_des': ROOT_PATH_LOCAL_MACHINE['root_machine'] + '/results/ADNI_des/'
}
