# Pose Representations for Deep Skeletal Animation

Official implementation of dual quaternion transformations as described in the paper [**"Pose Representations for Deep Skeletal Animation"**](https://nefeliandreou.github.io/assets/pdf/DQ_21.pdf).
> [Nefeli Andreou](nefeliandreou.github.io)

Please visit our [**project page**](https://nefeliandreou.github.io/projects/pose_representation/) for more details!

<p float="center">
  <img src="/assets/DQ_21.gif" width="100%" />
</p>

Check our YouTube video for [qualitative results](https://www.youtube.com/watch?v=bZKc_8s-XIk).


## Installation 
### 1. Create conda environment

```
conda env create -f environment.yml
conda activate deep_env
```
The code was tested on Python 3.6.9 and PyTorch 1.2.0. 

## Usage

- The files [extenddb.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/extenddb.py) and [generate_motion_in_dualquaternions.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/generate_motion_in_dualquaternions.py) are used to convert the .bvh files to the different representations. 
- Forward kinematics is calculated using the file [skeleton.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/common/skeleton.py).
- The file [dualquats.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/dualquats.py) contains the operations which are used during training (calculating translation/rotation, etc.)
- The file [twist_losses.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/twist_losses.py) contains the operations which are used during training of acRNN and is based on  [dualquats.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/dualquats.py). 

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

Note that the functions in [common](https://github.com/nefeliandreou/PoseRepresentation/tree/master/src/common) are borrowed by [QuaterNet](https://github.com/facebookresearch/QuaterNet), while the functions in [bvh](https://github.com/nefeliandreou/PoseRepresentation/tree/master/src/bvh) are borrowed by [acRNN](https://github.com/papagina/Auto_Conditioned_RNN_motion). 
