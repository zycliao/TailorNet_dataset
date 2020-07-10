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
[t-shirt_male](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_male.zip)\(6.9G\)  
[old-t-shirt_female](https://datasets.d2.mpi-inf.mpg.de/tailornet/old-t-shirt_female.zip)\(10G\)  
[t-shirt_female_sample](https://datasets.d2.mpi-inf.mpg.de/tailornet/t-shirt_female_sample.zip)\(19M\)
3. Specify the variable `DATA_DIR` in `global_var.py`  
4. Unzip all downloaded files to `DATA_DIR`  

## Dataset description
Currently, we have 5 garment classes (t-shirt, shirt, pant, skirt, old-t-shirt). 
In TailorNet paper, we trained and tested our model using `old-t-shirt`. 
Compared to `old-t-shirt`, `t-shirt` has a different topology, higher quality and larger style variation. 
Use `old-t-shirt` if you want a fair comparison with the results in our paper.  
   
The dataset structure looks like this:
```
DATA_DIR
----smpl
----apose.npy
----garment_class_info.pkl
----split_static_pose_shape.npz

----<garment_class>_<gender> (e.g., t-shirt_female)
--------pose/
------------<shape_idx>_<style_idx> (e.g., 000_023)
--------shape/
--------style/
--------style_shape/
--------avail.txt
--------pivots.txt
--------test.txt
--------style_model.npz
```

We provide `apose.npy`, `garment_class_info.pkl` and `split_static_pose_shape.npz` separately in `dataset_meta.zip`, and each `<garment_class>_<gender>` in a separate zip file.

- `split_static_pose_shape.npz` contains a dictionary `{'train': <train_idx>, 'test': <test_idx>}` where `<train_idx>` and `<test_idx>` are np arrays specifying the indices of poses which goes into train and test set respectively.
- `garment_class_info.pkl` contains a dictionary `{<garment_class>: {'f': <f>, 'vert_indices': <vert_indices>} }` where `<vert_indices>` denotes the vertex indices of high resolution SMPL body template which defines the garment topology of `<garment_class>`, and `<f>` denotes the faces of template garment mesh.
- `apose.npy` contains the thetas for A-pose on which garment style space is modeled.
- For each `<garment_class>_<gender>`,
  - `shape` directory contains uniformally chosen shape(beta) parameters.
  - `style_model.npz` contains a dictionary with these variables: `<pca_w>`, `mean`, `coeff_mean`, `coeff_range`. For given style `gamma`, garment vertices can be obtained using the following equation:
    - `pca_w * (gamma + coeff_mean) + mean`
  - `style` directory contains uniformally chosen style(gamma) parameters.
  - All styles are simulated on all shapes in A-pose and results are stored in `style_shape` directory. Out of those, shape_style pairs (also called pivots) with feasible simulation results are listed in `avail.txt`.
  - `pivots.txt` lists those pivots which are chosen as per the algorithm described in subsection - Choosing K Style-Shape Prototypes - to simulate training data. `test.txt` lists additional pivots chosen to generate testing data.
  - Each chosen pivot, denoted as `<shape_idx>_<style_idx>`, is simulated in few pose sequences. Simulation results are stored in `pose/<shape_idx>_<style_idx>` directory as unposed garment displacements. (Garment displacements are added on unposed template before applying standard SMPL skinning to get the final garment. See paper for details.)
  - `pose/<shape_idx>_<style_idx>` also contains displacements for smoothed unposed garment.

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
