#!/usr/bin/python

import xml.etree.ElementTree as ET

#------------------------------------------------------------------------------------------
# Function: to read meta-data
#------------------------------------------------------------------------------------------

def find_xml_file(data_params, subject_ID):
    import os
    for file in os.listdir(data_params['adni_1_meta_data']):
        if file.endswith(".xml") and str(subject_ID).lower() in file.lower():
            return read_xml_file(os.path.join(data_params['adni_1_meta_data'], file))
        
def read_xml_file(path_file):
    return ET.parse(path_file).getroot()

# get  [ID, Date, Class, Age, Sex, MMSE] 
def get_Subject_info(data_params, subject_ID):
    root = find_xml_file(data_params, subject_ID)
    # models

    _Date_Acquisition = root.findall('.//project/subject/study/series/dateAcquired')[0].text
    _Groupe_ = root.findall('.//project/subject/subjectInfo/[@item="DX Group"]')[0].text
    _AGE_ = root.findall('.//project/subject/study/subjectAge')[0].text
    _SEX_ = root.findall('.//project/subject/subjectSex')[0].text
    _MMSE_ = root.findall('.//project/subject/visit/assessment/[@name="MMSE"]/component/assessmentScore')[0].text    
    _GDS_ = root.findall('.//project/subject/visit/assessment/[@name="GDSCALE"]/component/assessmentScore')[0].text
    _CDR_ = root.findall('.//project/subject/visit/assessment/[@name="CDR"]/component/assessmentScore')[0].text
    

    _Groupe_ =  convert_class_name(_Groupe_)
    return [subject_ID, _Date_Acquisition, _Groupe_, _AGE_, _SEX_, _MMSE_, _GDS_, _CDR_]


# Convert class name
def convert_class_name(groupe):
    if 'MCI' in groupe:
        return "MCI"
    if 'Normal' in groupe:
        return "NC"
    if 'AD' in groupe:
        return "AD"
        