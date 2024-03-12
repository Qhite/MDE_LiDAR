#! /bin/bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cudatoolkit=11.7 -c pytorch -c nvidia -y
conda install fvcore iopath -c fvcore -c iopath -c conda-forge -y
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
conda install scipy opencv numpy pandas pyarrow matplotlib yacs tqdm -c conda-forge -y
pip install importlib-metadata flops-profiler seaborn