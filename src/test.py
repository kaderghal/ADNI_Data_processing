import xml.etree.ElementTree as ET



file_name = '/home/karim/workspace/ADNI_workspace/ADNI1/meta-data/ADNI_123_S_0088_MPR____N3__Scaled_S10795_I64400.xml'



tree = ET.parse(file_name)
root = tree.getroot()

Groupe = root.findall('.//project/subject/subjectInfo/[@item="DX Group"]')[0].text


MMSE = root.findall('.//project/subject/visit/assessment/[@name="MMSE"]/component/assessmentScore')[0].text

AGE = root.findall('.//project/subject/study/subjectAge')[0].text


SEX = root.findall('.//project/subject/subjectSex')[0].text




# Groupe = root.findall('.//project/subject/subjectInfo/[@item="DX Group"]')[0].text
print(SEX)
    
    
    
    