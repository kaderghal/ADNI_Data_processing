# ADNI_Data_processing

The Alzheimer's Disease Neuroimaging Initiative (ADNI) unites researchers with study data as they work to define the progression of Alzheimer's disease (AD). This repository includes the preprocessing of the data to extract 2D and 3D data with a specific  prepration to feed Neural Network for Alzheimer's Disease Classification problem.


# Repository organization

* dataset: external URl to download the daset useed for the preprocessing 
* sources: src code with python language.



## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pytorch 
```

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

* Python 3
# Requirements

1. Linux (Ubuntu Distribution : 18.04 LTS)
2. Python 3.6 (updated to python 3)
3. python packages:
..* nibabel
..* lmdb
..* scipy
..* pillow



3. Deep learning Framework
..* Caffe or
..* Tensorflow

### Performance of the models on Kinetics
- List of Acronyms:
This table shows the averaged accuracies over top-1 and top-5 on Kinetics.

| Abvs.| meaning                     |
|:---|:---:|
| AD   | Alzheimer's Disease         |
| MCI  | Mild Co,gnitive Impairment  |
| NC   | Normal Control              |
| MMSE | ...                         |
| sMRI | Structural Magnitic Imaging |
| DTI  | Diffusion Tensor Imaging    |
| HIPP | Hippocampus                 |



### Author
ADERGHAL KARIM 2019
LaBRI - University of Bordeaux - Bordeaux/France
LabSIV - University Ibn Zohr - Agadir/Morocco
email: {aderghal}.{karim}@gmail.com
email: {karim}.{aderghal}@labri.fr
[link text itself]: 
[http://www.labri.fr/perso/kadergha](http://www.labri.fr/perso/kadergha)


## Citation

If you use this code or pre-trained models, please cite the following:

```bibtex
@inproceedings{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6546--6555},
  year={2018},
}
```