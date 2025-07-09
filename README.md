# SSSG: Subsurface Scattering Gaussians for Surface Reconstruction of Translucent Objects

SSSG is a novel 3DGS-based surface reconstruction pipeline for translucent objects. It is realized based on PGSR.
The code is under organization.

## Installation

The repository contains submodules, thus please check it out with 
```shell
conda create -n sssg python=3.8
conda activate sssg

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install submodules/diff-plane-rasterization
pip install submodules/simple-knn
```

## Dataset Preprocess
We use NeuralTO Sync dataset: https://1drv.ms/f/c/20a93f3338af3e61/EopYJfbUPcFNk9WsRaN4HXkBdzG-ndPMd7JsCdI-gX-tjA?e=Aq59gV.
Download the dataset and save it at a folder like ./neuralto

The data folder should like this:
```shell
neuralto
|- gummybear
|  |- mask
|     |- 0000.png
|     |- 0001.png
|     |- ...
|  |- train
|     |- 0000.png
|     |- 0001.png
|     |- ...
|  |- test
|     |- 0000.png
|     |- 0001.png
|     |- ...
|  |- transforms_train.json
|  |- transforms_val.json
|
|- ...
```

#### Some Suggestions:
- Adjust the threshold for selecting the nearest frame in ModelParams based on the dataset;
- -r n: Downsample the images by a factor of n to accelerate the training speed;
- --max_abs_split_points 0: For weakly textured scenes, to prevent overfitting in areas with weak textures, we recommend disabling this splitting strategy by setting it to 0;
- --opacity_cull_threshold 0.05: To reduce the number of Gaussian point clouds in a simple way, you can set this threshold.

### Training
```
python train.py -s data_path[eg. ./neuralto/yuanbao] -m out_path
```

#### Some Suggestions:
- Adjust max_depth and voxel_size based on the dataset;
- --use_depth_filter: Enable depth filtering to remove potentially inaccurate depth points using single-view and multi-view techniques. For scenes with floating points or insufficient viewpoints, it is recommended to turn this on.
```shell
# Rendering and Extract Mesh
python render.py -m out_path --max_depth 10.0 --voxel_size 0.01
```

## Acknowledgements 


## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
```
