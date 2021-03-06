
## ADNI_Data_processing

The Alzheimer's Disease Neuroimaging Initiative (ADNI) unites researchers with study data as they work to define the progression of Alzheimer's disease (AD). This repository includes the preprocessing of the data to extract 2D and 3D data with a specific prepration and pytroch based-projet to work and feed Neural Network for Alzheimer's Disease Classification problem.

## Repository organization

* dataset: external URl to download the dataset used in the project
* sources: src folder contains two subfolder :
> - code source: with python language to prepare the datasets.
> - pytorch project: for creating architectures and training parameters  

## Requirements

* Linux Operating system (Ubuntu Distribution : 18.04 LTS)
* Python >= (3.6)
* python libraries:

## Install python virtual environment (optionnal)

```bash
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
sudo python3 get-pip.py
sudo pip install virtualenv virtualenvwrapper
sudo rm -rf ~/.cache/pip get-pip.py
nano .bashrc
```

> add these lines to the bashrc file  [by karim for python env]

```bash
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```

```bash
source .bashrc
mkvirtualenv ADNI_dl4cv -p python3
workon ADNI_dl4cv
pip3 list
```

### Install python libs 

> 1. for upgrading the setuptools & pip 
```bash
pip3 install --upgrade setuptools pip3
```

> 2. used to plot graphes and images
```bash
pip3 install matplotlib
```

> 3. to check python syntaxe ...
```bash
pip3 install pylint pyparsing six
```

> 4. nibabel (for NIfTI Medical images)
```bash
> pip3 install nibabel 
```

> 5. for scientific data structure (fast matrix & array ) 
```bash
> pip3 install numpy
> pip3 install scipy 
```

> 6. work on images
```bash
> pip3 install pillow
```

### Deep learning Framework
*  [PyTorch](http://pytorch.org/)
```bash
pip3 install pytorch # for cpu installation (see Official website)
```

pytorch website : [https://pytorch.org/](https://pytorch.org/)




## List of Acronyms:

This table shows the acronyms used in the project.

| Abvs.| meaning |
|:---|:---:|
| AD | Alzheimer's Disease |
| MCI | Mild Co,gnitive Impairment |
| NC | Normal Control |
| MMSE | Mini-Mental State Examination |
| sMRI | Structural Magnitic Imaging |
| DTI | Diffusion Tensor Imaging |
| HIPP | Hippocampus |
| PPC | Posterior Parietal Cortex |

### Author Information

  > ADERGHAL KARIM 2020

LaBRI - University of Bordeaux - Bordeaux/France

LabSIV - University Ibn Zohr - Agadir/Morocco

email: {aderghal}.{karim}@gmail.com

email: {karim}.{aderghal}@labri.fr

page: [http://www.labri.fr/perso/kadergha](http://www.labri.fr/perso/kadergha)

## Citation


