# Road Profile Estimation
This README file gives only a short introduction into the full road profile estimation prototype of the corresponding research project.

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
