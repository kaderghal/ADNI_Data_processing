import xml.etree.ElementTree as ET





########## XML ###################""
# sex, age, mmse
###############################
def find_xml_file(data_params, subject_ID):
    import os
    for file in os.listdir(data_params['adni_1_meta_data']):
        if file.endswith(".xml") and str(subject_ID).lower() in file.lower():
            return read_xml_file(os.path.join(data_params['adni_1_meta_data'], file))
        

def read_xml_file(path_file):
    return ET.parse(path_file).getroot()



def get_Subject_info(data_params, subject_ID):
    root = find_xml_file(data_params, subject_ID)

    # models
    _Groupe_ = root.findall('.//project/subject/subjectInfo/[@item="DX Group"]')[0].text
    _MMSE_ = root.findall('.//project/subject/visit/assessment/[@name="MMSE"]/component/assessmentScore')[0].text
    _AGE_ = root.findall('.//project/subject/study/subjectAge')[0].text
    _SEX_ = root.findall('.//project/subject/subjectSex')[0].text
    
    return [_Groupe_, subject_ID, _SEX_, _AGE_, _MMSE_]