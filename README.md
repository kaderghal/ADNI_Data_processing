# ADNI_Data_processing

The Alzheimer's Disease Neuroimaging Initiative (ADNI) unites researchers with study data as they work to define the progression of Alzheimer's disease (AD). This repository includes the preprocessing of the data to extract 2D and 3D data with a specific prepration and pytroch based-projet to work and feed Neural Network for Alzheimer's Disease Classification problem.


# Repository organization

* dataset: external URl to download the dataset used in the project 
* sources: src folder contains two subfolder :
- code source with python language to prepare the datasets.
- pytorch project: for creating architectures and training parameters ....


## Requirements

* Linux Operating system (Ubuntu Distribution : 18.04 LTS)
* Python 2.7 (w'll move to 3.6)

* python libraries:

1. nibabel (for niffty Medical images)
2. scipy, numpy
3. pillow, 

# Deep learning Framework

* [PyTorch](http://pytorch.org/)

```bash
pip install pytorch 
```

# List of Acronyms:
This table shows the acronyms used in the project.

| Abvs.| meaning                     |
|:---|:---:|
| AD   | Alzheimer's Disease         |
| MCI  | Mild Co,gnitive Impairment  |
| NC   | Normal Control              |
| MMSE | ...                         |
| sMRI | Structural Magnitic Imaging |
| DTI  | Diffusion Tensor Imaging    |
| HIPP | Hippocampus                 |
| PPC | ....                 |



### Author
ADERGHAL KARIM 2019
LaBRI - University of Bordeaux - Bordeaux/France
LabSIV - University Ibn Zohr - Agadir/Morocco
email: {aderghal}.{karim}@gmail.com
email: {karim}.{aderghal}@labri.fr
[link text itself]: 
[http://www.labri.fr/perso/kadergha](http://www.labri.fr/perso/kadergha)


## Citation

```bibtex
@inproceedings{hara3dcnns,
  author={Karim Aderghal and Karim Afdel and Jenny Benois-Pineau and Gwënaelle Catheline},
  title={3D Enhancing Siamese Network for Alzheimer's Disease Classification ....},
  booktitle={Proceedings of XXX},
  pages={1--8},
  year={2020},
}
```