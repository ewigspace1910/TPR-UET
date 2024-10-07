# Combination of Multiple Angles of View and Enhanced Pseudo-Label Generation for Unsupervised Person Re-Identification(MAPLE)
Official PyTorch implementation of [Combination of Multiple Angles of View and Enhanced Pseudo-Label Generation for Unsupervised Person Re-Identification]().

## Updates
- [10/2024] Initilized Repo.

## Overview

>In recent years, Person Re-Identification (ReID) has emerged as a vital research area, significantly impacting applications across diverse domains such as security and education. This study addresses the challenges inherent in ReID, primarily focusing on the shift from traditional supervised methods to unsupervised techniques. The latter can effectively manage the large data volumes and labor-intensive labeling required by supervised approaches. Unsupervised ReID is categorized into Unsupervised Domain Adaptation (UDA) and fully Unsupervised Learning (USL) strategies. Despite their potential, these strategies face challenges like dependency on annotated source domains or inaccuracies in pseudo-labeling. To address these issues, the paper introduces a novel TPR method that enhances the accuracy of USL models in person ReID. Our contributions include leveraging both global and local feature information, optimizing the DBSCAN clustering process, and integrating domain adaptation techniques to improve performance. Extensive experiments conducted on benchmarks Market-1501 and MSMT17 demonstrate that our approach surpasses several state-of-the-art methods.
## Getting Started

### Installation
```shell
download code.zip 
cd MAPLE
pip install -r requirement
```
### Preparing Datasets
```shell
mkdir /data
```
Download the object re-ID datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565) to `/data`.
The directory should look like:
```
/data
├── Market-1501-v15.09.15
└── MSMT17_V2
```
## Training
We utilize 2 A100 GPUs for training.
We use 384x128 sized images for Market-1501 and MSMT17.



## Acknowledgement
Some parts of the code is borrowed from [SpCL](https://github.com/yxgeee/SpCL).

## Citation
If you find this code useful for your research, please consider citing our paper:

````BibTex

````
