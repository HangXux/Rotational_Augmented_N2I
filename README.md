# Rotational Augmented Noise2Inverse

This repository provides basic codes for reproducing RAN2I on simulated data.

## Environment

### Conda environment:
```
conda create -n RAN2I python=3.8
conda activate RAN2I
conda install -c astra-toolbox astra-toolbox
pip install -r reqirements.txt
```

## Dataset



## Workflow

### Convert Dicom data to numpy data
Download dicom data to folder `dicomfile` with the structure as below:

    dicomfile
    ├── convert_dicom_to_npy.py
    └── Training_Image_Data_B30
        ├── L067
        │   ├── quarter_3mm
        │   │       ├── L067_QD_3_1.CT.0002.0001 ~ .IMA
        │   │       ├── L067_QD_3_1.CT.0002.0002 ~ .IMA
        │   │       └── ...
        │   └── full_3mm
        │           ├── L067_FD_3_1.CT.0002.0001 ~ .IMA
        │           ├── L067_FD_3_1.CT.0002.0002 ~ .IMA
        │           └── ... 
        ...
        │
        └── L506
            ├── quarter_3mm
            │       └── ...
            └── full_3mm
                    └── ...     

Run `convert_dicom_to_npy.py` to obtain normalized numpy data

### Split the sinograms and geometry

Run `split_preprocess.py` to split the sinograms and geometry angularly 
based on [ASTRA Toolbox](https://www.astra-toolbox.com/).
The split data are used for training.

### Train and test a model

Run `train.py` to train a new model. 

Run `test.py` to evaluate the pre-trained models.
We provide two pre-trained models in folder `weights`, 
which were obtained with different sampling views (512 and 1024) 
and the same photon counts (1e4) in the training. 
We also provide one preprocess split test data in folder `data`.




