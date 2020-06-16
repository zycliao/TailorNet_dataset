# TailorNet Dataset
This repository is a toolbox to process, visualize the dataset for "TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style" (CVPR 2020 Oral)  

[[model repository](https://github.com/chaitanya100100/TailorNet)][[arxiv](https://arxiv.org/abs/2003.04583)][[project website](https://virtualhumans.mpi-inf.mpg.de/tailornet/)][[YouTube](https://www.youtube.com/watch?v=F0O21a_fsBQ)]

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
[t-shirt_female](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female.zip)\(6.8G\)  
[t-shirt_male](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_male.zxip)\(6.9G\)  
[old-t-shirt_female](https://datasets.d2.mpi-inf.mpg.de/tailornet/old-t-shirt_female.zip)\(10G\)  
[t-shirt_female_sample](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female_sample.zip)\(19M\)
3. Specify the variable `DATA_DIR` in `global_var.py`  
4. Unzip all downloaded files to `DATA_DIR`  

## Dataset description
Currently, we have 5 garment classes (t-shirt, shirt, pant, skirt, old-t-shirt). 
In TailorNet paper, we trained and tested our model using `old-t-shirt`. 
Compared to `old-t-shirt`, `t-shirt` has a different topology, higher quality and larger style variation. 
Use `old-t-shirt` if you want a fair comparison with the results in our paper.  

In each (garment_class, gender) sub-dataset, all feasible (shape, style) combinations are in `avail.txt`.
All of them are simulated in A-pose and the results are in `style_shape/`.
Shape and style parameters can be accessed in `shape/` and `style/`.

Pivot (shape, style)s are recorded in `pivots.txt` and test (shape, style)s are in `test.txt`.
Each of them is simulated in multiple poses and the results are in `pose/{shape}_{style}`.

   
The dataset structure looks like this:
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
--------test.txt
----apose.npy
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

## TODO
- [ ] Dataset generation codes
- [ ] Style space visualizer
- [ ] Blender visualizer
- [ ] Google Drive/BaiduYun
- [ ] Shirt, pants, skirt
- [x] T-shirt
- [x] Basic visualizer

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
