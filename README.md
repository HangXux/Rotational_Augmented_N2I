# Rotational Augmented Noise2Inverse

This repository provides basic codes for reproducing RAN2I on simulated CT data.

## Environment

### Conda environment:
```
conda create -n RAN2I python=3.8
conda activate RAN2I
conda install -c astra-toolbox astra-toolbox
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r reqirements.txt
```

## Dataset
The data for simulation are from [AAPM dataset](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/folder/145241239366). We used full-dose images and [ASTRA Toolbox](https://www.astra-toolbox.com/) to simulate low-dose acquisition.


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

`physics.py` shows the process of splitting the sinograms and geometry angularly 
based on [ASTRA Toolbox](https://www.astra-toolbox.com/).

### Evaluate RAN2I

Run `python demo_RAN2I.py` in the terminal to evaluate a pre-trained RAN2I model. 

We provide two pre-trained models in folder `weights`, 
which were obtained with different sampling views (512 and 1024) 
and the same photon counts (1e4) in the training. 
We also provide one preprocess test data in folder `data`.




