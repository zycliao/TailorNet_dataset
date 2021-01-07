## usage
Note: Since our raw data is not public, scripts in this directory are only for reference.
If you want to model style space yourself, you might skip `copy_raw.py` and start from line 45 of `proc_raw.ppy`.
Also make sure you understand all parameters in `style_pca.py` and `gender_pca.py` and set them yourself, 
because those parameters may be highly dependent of raw data.  
1. copy_raw.py
copy registration meshes and do some processing
2. proc_raw.py
3. vis_raw.py
visualize raw/smooth garments
4. manually pick out noisy garments
5. style_pca.py
6. pca_interactive.py
adjust pca and trans for each gender.
Hardcode the parameter range in gender_pca.py
7. gender_pca.py

## extending skirt space
after running the above procedure,
rename the following files/folders:
skirt_reg->skirt_orig_reg
skirt_female->skirt_orig_female
pca/...skirt...->pca/...skirt_orig...
raw_data/...skirt...->raw_data/...skirt_orig...

then, run
proc_raw_skirt_ext.py  
style_pca.py  
gender_pca.py  
