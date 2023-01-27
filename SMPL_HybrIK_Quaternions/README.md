## Installation instructions

``` bash
# 1. Create a conda virtual environment.
conda create -n hybrik python=3.8 -y
conda activate hybrik

# 2. Install required modles
pip install -r requirements

# 3. Install PyTorch3D
curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz
export CUB_HOME=$PWD/cub-1.10.0
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# 4. Install HybrIK
cd HybrIK ; python setup.py develop  # or "pip install -e ."
```

## Download models
* Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) at `common/utils/smplpytorch/smplpytorch/native/models`.
* Download pretrained model (with predicted camera) from [ [Google Drive](https://drive.google.com/file/d/16Y_MGUynFeEzV8GVtKTE5AtkHSi3xsF9/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1kHTKQEKiPnrAKAUzOD-Xww) (code: `4qyv`) ].
