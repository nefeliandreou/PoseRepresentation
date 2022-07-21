# Pose Representations for Deep Skeletal Animation

Official implementation of dual quaternion transformations as described in the paper [**"Pose Representations for Deep Skeletal Animation"**](https://nefeliandreou.github.io/assets/pdf/DQ_21.pdf).
[![report](https://img.shields.io/badge/arXiv-2204.08451-b31b1b.svg)](https://arxiv.org/abs/2204.08451)
<a href="https://evonneng.github.io/learning2listen/"><img src="https://img.shields.io/badge/project page-github.io-blue"/></a> 
Please visit our [project page](https://nefeliandreou.github.io/projects/pose_representation/) for more details and check our [YouTube video](https://www.youtube.com/watch?v=bZKc_8s-XIk)!

<p float="center">
  <img src="/assets/DQ_21.gif" width="100%" />
</p>

## Installation 
Create conda environment

```
conda env create -f environment.yml
conda activate dq_env
```
The code was tested on Python 3.6.9 and PyTorch 1.2.0. 

## Usage

- [extenddb.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/extenddb.py) and [generate_motion_in_dualquaternions.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/generate_motion_in_dualquaternions.py) are used to convert the .bvh files to the different representations. 
- Forward kinematics is calculated using the  [skeleton.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/common/skeleton.py).
- [dualquats.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/dualquats.py) contains the operations which are used during training (calculating translation/rotation, etc.)
- [twist_losses.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/twist_losses.py) contains the operations which are used during training of acRNN and is based on  [dualquats.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/dualquats.py). 


## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that the functions in [common](https://github.com/nefeliandreou/PoseRepresentation/tree/master/src/common) are borrowed by [QuaterNet](https://github.com/facebookresearch/QuaterNet), the functions in [bvh](https://github.com/nefeliandreou/PoseRepresentation/tree/master/src/bvh) are borrowed by [acRNN](https://github.com/papagina/Auto_Conditioned_RNN_motion), 
and the functions in [DualQuaternion2.py](https://github.com/nefeliandreou/PoseRepresentation/blob/master/src/DualQuaternion2.py) from [this](https://github.com/Achllle/dual_quaternions/blob/master/src/dual_quaternions/dual_quaternions.py) repository. Please respect the individual licenses when using these files.

## Acknowledgments

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 860768.

## Citation
If you find this code useful in your research, please cite:
```
@misc{Andreou:2021:PoseRepresentation,
    author = {Andreou, Nefeli and Aristidou, Andreas and Chrysanthou, Yiorgos},
    title = {Pose Representations for Deep Skeletal Animation},
    eprint={2111.13907},
    year  = {2021},
    archivePrefix={arXiv}
}
```

