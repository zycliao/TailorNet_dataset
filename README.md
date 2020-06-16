# TailorNet
Official repository of "TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style" (CVPR 2020 Oral)  
[[arxiv](https://arxiv.org/abs/2003.04583)][[project website](https://virtualhumans.mpi-inf.mpg.de/tailornet/)][[Youtube](https://www.youtube.com/watch?v=F0O21a_fsBQ)]

## Requirements
python3  
pytorch  
chumpy  
opencv-python  
cython  

## SMPL model
1. Register and download SMPL models [here](https://smpl.is.tue.mpg.de/en)  
2. Unzip `SMPL_python_v.1.0.0.zip` and put `smpl/models/*.pkl` in `DATA_DIR/smpl`(specify `DATA_DIR` in `global_var.py`)   
3. Run `smpl_lib/convert_smpl_models.py`  

## Data preparation
1. Download meta data of the dataset  
[dataset_meta](https://datasets.d2.mpi-inf.mpg.de/tailornet/dataset_meta.zip)
2. Download one or more sub-dataset (other garment classes are coming soon)   
[t-shirt_female](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female.zip)\(6.6G\)  
[t-shirt_male](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_male.zip)\(6.9G\)  
[t-shirt_female_sample](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female_sample.zip)\(19M\)
3. Specify the variable `DATA_DIR` in `global_var.py`  
4. Unzip all downloaded files to `DATA_DIR`  
5. The dataset structure looks like this:
```
DATA_DIR
----smpl
----{garment_class}_{gender} (e.g., t-shirt_female)
--------pose
------------{shape}_{style} (e.g., 000_023)
--------shape
--------style
--------style_shape
--------avail.txt
--------pivots.txt
----apose.pkl
----garment_class_info.pkl
----split_static_pose_shape.npz
```
  

## Visualize the dataset
1. Install the renderer
```
cd render_lib
python setup.py build_ext -i
```
2. Run the visualizer
```
python visualize_dataset.py
```

## Training
Coming soon

## Citation
Cite us:
```
@inproceedings{patel20tailornet,
        title = {TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style},
        author = {Patel, Chaitanya and Liao, Zhouyingcheng and Pons-Moll, Gerard},
        booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {jun},
        organization = {{IEEE}},
        year = {2020},
    }
```
