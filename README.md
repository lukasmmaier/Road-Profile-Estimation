# 3D Road Profile Estimation based on Monocular Vision
This GitHub repository is part of my research project at the IAS (University of Stuttgart) for my M.Sc. in Electromobility. It investigates whether monocular vision can be used for real-time 3D estimation of road profiles. Since the road profile significantly affects passenger comfort and vehicle dynamics, its estimation is crucial for active suspension systems allowing to improve both aspects. The developed concept combines a Visual Odometry approach (i.e. [DVSO](https://arxiv.org/abs/1807.02570)) with a dense reconstruction method (i.e. [MonoRec](https://arxiv.org/abs/2011.11814)), followed by a post processing module. The evaluation showed that the upcoming road profile can be estimated using monocular vision. An example is given in the following image.

![Example. Prototype Input - Prototype Output.](/pictures/example.png)

This README file gives only a short introduction into the full 3D road profile estimation prototype. For more information, the corresponding thesis describes the concept and the prototype in detail.

## Prototype System Architecture:
The conceptual system should consist of 3 modules: 

<strong>DVSO => MonoRec => Post Processing</strong>

The implemented prototype consists of 2 modules, compensating for DVSO by using information from the dataset:

<strong>MonoRec => Post Processing</strong>

This repository only contains the Post Processing part. For evaluation and testing, the KITTI odometry dataset is used.

### DVSO:
- Paper: https://arxiv.org/abs/1807.02570
- Unofficial GitHub Repository: https://github.com/SenZHANG-GitHub/DVSO

### MonoRec:
- Paper: https://arxiv.org/abs/2011.11814
- Official GitHub Repository: https://github.com/Brummi/MonoRec

### KITTI Odometry Dataset:
- Paper: https://ieeexplore.ieee.org/document/6248074
- Website/Download: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

## Installation
The full installation manual and user manual can be found in the corresponding thesis.

## Configuration
To configure the algorithms output, use the ```config.gin``` file located under the ```configs``` folder.

To switch between regular or evaluation mode, manually set the FLAG (global variable) in line ```12``` of the ```main.py``` to ```true``` or ```false```.

## Test
Added test files (```test/MonoRec/saved/pointclouds/monorec/seequence_05_60-120_256.ply``` and ```test/MonoRec/data/dataset/poses_dvso/05.txt```) for testing usage without MonoRec. The ```MonoRec``` folder inside the ```test```folder should be placed next to this repository.
