# PVRNet
PVRNet: Point-View Relation Neural Network for 3D Shape Recognition (AAAI 2019)

Created by Haoxuan You, Yifan Feng, Xibin Zhao, Changqing Zou, Rongrong Ji, Yue Gao from Tsinghua University.

![](https://github.com/iMoonLab/PVRNet/blob/master/docs/pipeline.png)
### Introduction
This work will appear in AAAI 2019. We propose a point-view relation neural network called PVRNet for 3D shape recognition and retrieval. You can chekc our [paper](https://arxiv.org/abs/1812.00333) for more details.

In this repository, our code and data are released for training our PVRNet on ModelNet40 dataset.

### Citation
If you find our work useful in your research, please cite our paper:
```
@article{you2018pvrnet,
title={PVRNet: Point-View Relation Neural Network for 3D Shape Recognition},
author={You, Haoxuan and Feng, Yifan and Zhao, Xibin and Zou, Changqing and Ji, Rongrong and Gao, Yue},
journal={AAAI 2019},
year={2018}
}
```
### Configuration
Code is tested under the environment of Pytorch 0.4.1, Python 3.6 and CUDA 9.0 on Ubuntu 16.04. 

Data: [point cloud data](https://drive.google.com/file/d/1DUh_8PQjh3ds4yO0O8q_vb0HPistOJ4y/view?usp=sharing) and [multi-view(12-view) data](https://drive.google.com/file/d/12JbIPLvcSUsMjxb_CZYXI8xQK2UKosio/view?usp=sharing) from ModelNet40 dataset.

Pretrained Model: [multi-view part(MVCNN)](https://drive.google.com/file/d/1dZG7XojtPS9Cl5aaH4iWXA_N2PximB6i/view?usp=sharing), [point cloud part(DGCNN)](https://drive.google.com/file/d/1fY9E44xuPwUFxJ_BIeP5NXwrB7DQm1tw/view?usp=sharing) and [PVRNet](https://drive.google.com/file/d/1g3Ef68jRSY2mNf54dOeqNFYZTm4cO13d/view?usp=sharing)  

### Usage
+ Download data and pretrained ckpt from above links. Create dir for data as well as result, and place them under corresponding dirs(./data/ and ./result/ckpt/).

    ```mkdir -p data result/ckpt```
    
+ Train PVRNet. This would use pretrained MVCNN model and DGCNN model saved in ./result/ckpt:

    ``` python train_pvrnet.py```

+ If validate the performance of PVRNet with our pretrained model:

    `python val_pvrnet.py`

    If validate the performance of pretrained MVCNN and DGCNN models:
    ```
    python val_mvcnn.py
    python val_pc.py
    ```

+ If you want to train new model for MVCNN and DGCNN:

    
    ```
    python train_mvcnn.py
    python train_pc.py
    ```


### License
Our code is released under MIT License (see LICENSE file for details).


    
