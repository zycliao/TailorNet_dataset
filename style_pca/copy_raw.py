"""
Go through the scanning dataset and find available registrations for each garment class
save their displacement (in one .npy file) and meta information (triangulation, mapping with SMPL)
INPUT: registration mesh under /BS/bharat-2/work/data/renderings
OUTPUT: {ROOT}/raw_data/{garment_class}.npy
        {ROOT}/garment_class_info.pkl
"""
import os
import json
import numpy as np
import pickle
from utils.rotation import get_Apose
import trimesh

APOSE = get_Apose()

upper_lower_dict = {'t-shirt': 'UpperClothes', 'pant': 'Pants', 'shirt': 'UpperClothes', 'short-pant': 'Pants'
                    }
v_num = {'t-shirt': 7702, 'shirt': 9723, 'pant': 4718, 'short-pant': 2710, 'coat': 10116, 'skirt': 7130}


def sort_types(RAW_DIR):
    set_name = os.listdir(RAW_DIR)
    people_name = {}
    garment_types = {}
    garment_paths = {}
    genders = {}
    for sn in set_name:
        names = []
        for n in os.listdir(os.path.join(RAW_DIR, sn)):
            people_dir = os.path.join(RAW_DIR, sn, n)
            if not os.path.isdir(people_dir):
                continue
            json_path = os.path.join(people_dir, "{}_annotations.json".format(n))
            if not os.path.exists(json_path):
                continue
            with open(json_path) as jf:
                annotations = json.load(jf)
            gtypes = annotations['garments']

            # statistics
            for gt in gtypes:
                if gt not in garment_types:
                    garment_types[gt] = 1
                    garment_paths[gt] = [people_dir]
                else:
                    garment_types[gt] += 1
                    garment_paths[gt].append(people_dir)

            gender = annotations['gender']
            if gender not in genders:
                genders[gender] = 1
            else:
                genders[gender] += 1

    print(garment_types)
    print(genders)
    return garment_paths


if __name__ == '__main__':
    garment_class = 'short-pant'
    upper_lower = upper_lower_dict[garment_class]
    RAW_DIR = '/BS/bharat-2/work/data/renderings'
    SAVE_DIR = '/BS/cloth-anim/static00/tailor_data/raw_data'

    garment_paths = sort_types(RAW_DIR)
    garment_paths = garment_paths[garment_class]

    all_disp = []
    betas = []
    vert_indices, faces = None, None
    for garment_path in garment_paths:
        people_name = os.path.split(garment_path)[1]
        garment_ply_path = os.path.join(garment_path, 'temp20', '{}_unposed.ply'.format(upper_lower))
        garment_pkl_path = os.path.join(garment_path, 'temp20', '{}_unposed.pkl'.format(upper_lower))
        if not (os.path.exists(garment_ply_path) and os.path.exists(garment_pkl_path)):
            continue
        with open(garment_pkl_path, 'rb') as f:
            garment_info = pickle.load(f, encoding='latin-1')

        displacement = garment_info['garment_offsets']
        beta = garment_info['betas']
        assert beta.shape == (10,)
        if len(displacement) != v_num[garment_class]:
            continue
        all_disp.append(displacement)
        betas.append(beta)
    all_disp = np.array(all_disp).astype(np.float32)
    np.save(os.path.join(SAVE_DIR, '{}.npy'.format(garment_class)), all_disp)
    betas = np.array(betas).astype(np.float32)
    np.save(os.path.join(SAVE_DIR, '{}_betas.npy'.format(garment_class)), betas)

