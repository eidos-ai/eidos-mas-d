## Installation instructions

``` bash
# 1. Create a conda virtual environment.
conda create -n hybrik python=3.8 -y
conda activate hybrik

# 2. Install required modles
pip install -r requirements

# 3. Install PyTorch3D
pip install git+ssh://git@github.com/facebookresearch/pytorch3d.git@stable

# 4. Install HybrIK
cd HybrIK ; python setup.py develop  # or "pip install -e ."
```

## Download models
* Download the SMPL model `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` from [here](https://smpl.is.tue.mpg.de/) at `common/utils/smplpytorch/smplpytorch/native/models`.
* Download our pretrained model (paper version) from [ [Google Drive](https://drive.google.com/file/d/1SoVJ3dniVpBi2NkYfa2S8XEv0TGIK26l/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/13rPFHO6FWoy7DK066XY1Fw) (code: `qre2`) ].
* Download our pretrained model (with predicted camera) from [ [Google Drive](https://drive.google.com/file/d/16Y_MGUynFeEzV8GVtKTE5AtkHSi3xsF9/view?usp=sharing) | [Baidu](https://pan.baidu.com/s/1kHTKQEKiPnrAKAUzOD-Xww) (code: `4qyv`) ].
