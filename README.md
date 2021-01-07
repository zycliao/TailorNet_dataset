# TailorNet Dataset
This repository is a toolbox to process, visualize the dataset for "TailorNet: Predicting Clothing in 3D as a Function of Human Pose, Shape and Garment Style" (CVPR 2020 Oral)  

[[model repository](https://github.com/chaitanya100100/TailorNet)][[arxiv](https://arxiv.org/abs/2003.04583)][[project website](https://virtualhumans.mpi-inf.mpg.de/tailornet/)][[YouTube](https://www.youtube.com/watch?v=F0O21a_fsBQ)]

## Update
2021/1/7    data generation codes  
2020/12/7   short pants, skirt are available  
2020/7/31   pants, shirt are available


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
All data is available [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/dQCHykYo77EQYS8)
1. Download meta data (dataset_meta.zip) of the dataset  

2. Download one or more sub-dataset (other garment classes are coming soon)   
t-shirt_female(6.9G)  
t-shirt_male(7.2G)  
old-t-shirt_female(10G)  
t-shirt_female_sample(19M)  
shirt_female(12.7G)  
shirt_male(13.5G)  
pant_female(3.3G)  
pant_male(3.4G)  
short-pant_female(1.9G)  
short-pant_male(2G)  
skirt_female(5G)

3. Specify the variable `DATA_DIR` in `global_var.py`  
4. Unzip all downloaded files to `DATA_DIR`  

## Dataset Description
Currently, we have 6 garment classes (t-shirt, shirt, pant, skirt, short-pant, old-t-shirt). 
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
  
  - `style_model.npz` contains a dictionary with these variables: `pca_w`, `mean`, `coeff_mean`, `coeff_range`. For given style `gamma`, garment vertices can be obtained using the following equation:
    - `pca_w * (gamma + coeff_mean) + mean`
  - `style` directory contains uniformally chosen style(gamma) parameters.
  - All styles are simulated on all shapes in A-pose and results are stored in `style_shape` directory. Out of those, shape_style pairs (also called pivots) with feasible simulation results are listed in `avail.txt`.
  - `pivots.txt` lists those pivots which are chosen as per the algorithm described in subsection - Choosing K Style-Shape Prototypes - to simulate training data. `test.txt` lists additional pivots chosen to generate testing data.
  - Each chosen pivot, denoted as `<shape_idx>_<style_idx>`, is simulated in few pose sequences. Simulation results are stored in `pose/<shape_idx>_<style_idx>` directory as unposed garment displacements. (Garment displacements are added on unposed template before applying standard SMPL skinning to get the final garment. See paper for details.)
  - `pose/<shape_idx>_<style_idx>` also contains displacements for smoothed unposed garment.

## Visualize the dataset
1. Install the renderer
```
cd utils/render_lib
python setup.py build_ext -i
```
2. Run the visualizer
```
python visualize_dataset.py
```

## Dataset Generation
Please check readme.md in each directory for detail  
1. style_pca  
Scripts that process garment registrations and model the garment style space.  
2. simulation_style  
Simulate all (style, shape) combinations in A-pose.  
3. pivots  
Generate pivots and test set.  
4. simulation_pose  
Simulate different poses for pivots and test (style, shape).  

Since our raw data is not public and simulation in Marvelous Designer cannot be scripted,
these codes are only for reference. If you want to simulate your own data, 
make sure you understand most code and the paper, 
so that you can modify parameters that are highly dependent of the data.

## Count the dataset
```
python count_data.py
——————————————————————————————————————————————————————————————————————————————
|          |          |    train style_shape|     test style_shape|          |
|     class|    gender|train pose| test pose|train pose| test pose|     total|
——————————————————————————————————————————————————————————————————————————————
|   t-shirt|    female|     14589|      3309|       776|       224|     18898|
|   t-shirt|      male|     14397|      3353|       815|       185|     18750|
|     shirt|    female|     14553|      3342|       856|       144|     18895|
|     shirt|      male|     14322|      3328|       831|       169|     18650|
|      pant|    female|     14569|      3430|       805|       195|     18999|
|      pant|      male|     14562|      3423|       793|       203|     18981|
|short-pant|    female|     14546|      3451|       804|       196|     18997|
|short-pant|      male|     14563|      3426|       796|       203|     18988|
|     skirt|    female|     14554|      3444|       803|       197|     18998|
|                total|    130655|     30506|      7279|      1716|    170156|
——————————————————————————————————————————————————————————————————————————————
```

## TODO
- [x] Dataset generation codes
- [x] Style space visualizer
- [ ] Blender visualizer
- [x] Shirt, pants, skirt
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
