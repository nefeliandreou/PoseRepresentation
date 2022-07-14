# Pose Representations for Deep Skeletal Animation

Official implementation of dual quaternion transformations as described in the paper [**"Pose Representations for Deep Skeletal Animation"**](https://nefeliandreou.github.io/assets/pdf/DQ_21.pdf).

Please visit our [**project page**](https://nefeliandreou.github.io/projects/pose_representation/) for more details!

The code was tested on Python 3.6.9 and PyTorch 1.2.0. 

## Installation 
### 1. Create conda environment

```
conda env create -f environment.yml
conda activate deep_env
```

## Instructions
### 1. For acRNN 

### 2. For QuaterNet
The transformations used for the [**QuaterNet**](https://github.com/facebookresearch/QuaterNet) experiments can be found in $src_QuaterNet. 

#### Bibtex
If you find this code useful in your research, please cite:
```
@misc{Andreou:2021:DQ,
    author = {Andreou, Nefeli and Aristidou, Andreas and Chrysanthou, Yiorgos},
    title = {A Hierarchy-Aware Pose Representation for Deep Character Animation},
    eprint={2111.13907},
    year  = {2021},
    archivePrefix={arXiv}
}
```

## License
This code is distributed under an [MIT LICENSE](LICENSE).


