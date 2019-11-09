

AUTHOR_INFO = {
    'name': 'Alz_ADNI_process',
    'version': '1.2',
    'year': '2019',
    'description': 'Extracting data for CNN Alzheimer\'s Disease Classification',
    'url': 'http://github.com/kaderghal',
    'author': 'Karim ADERGHAL',
    'email': 'aderghal.karim@gmail.com',
    'university': 'Bordeaux',
    'lab': 'LaBRI'
}

def get_author_info():
    tempo_dict = {}
    tempo_dict['name'] = str(AUTHOR_INFO['name'])
    tempo_dict['version'] = str(AUTHOR_INFO['version'])
    tempo_dict['year'] = str(AUTHOR_INFO['year'])
    tempo_dict['description'] = str(AUTHOR_INFO['description'])
    tempo_dict['url'] = str(AUTHOR_INFO['url'])
    tempo_dict['author'] = str(AUTHOR_INFO['author'])
    tempo_dict['email'] = str(AUTHOR_INFO['email'])
    tempo_dict['lab'] = str(AUTHOR_INFO['lab'])
    return tempo_dict



def print_author_info():
    print("Author Information :\n")
    for k, v in get_author_info().items():
        print('\t[' + k + ']: ' + v)
    print ("\n")
    
    
# print("A: ", get_author_info().items())
print_author_info()