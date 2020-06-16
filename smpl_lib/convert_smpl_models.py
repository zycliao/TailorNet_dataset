"""
Save SMPL model weights in numpy format
"""
import os
import numpy as np
import pickle as pkl
from smpl_lib.smpl_paths import SmplPaths
import global_var


def proc_data(k, v):
    if k == 'J_regressor':
        v = v.todense()
    if k == 'kintree_table':
        return np.array(v)[0].astype(np.int)
    else:
        return np.array(v, dtype=np.float32)

if __name__ == '__main__':
    genders = ['male', 'female']
    keys = ['f', 'v_template', 'shapedirs', 'J_regressor', 'posedirs',
            'kintree_table', 'weights', 'J']
    SAVE_DIR = os.path.join(global_var.DATA_DIR, 'smpl')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    for gender in genders:
        low_res = SmplPaths(gender=gender)
        low_res_model = pkl.load(open(low_res.get_smpl_file(), 'rb'), encoding='latin1')
        print(low_res_model.keys())
        high_res_model = low_res.get_hres_smpl_model_data()
        low_res_dict, high_res_dict = {}, {}
        for k in keys:
            low_res_dict[k] = proc_data(k, low_res_model[k])
            high_res_dict[k] = proc_data(k, high_res_model[k])
        np.savez(os.path.join(SAVE_DIR, 'smpl_{}.npz'.format(gender)), **low_res_dict)
        np.savez(os.path.join(SAVE_DIR, 'smpl_hres_{}.npz'.format(gender)), **high_res_dict)