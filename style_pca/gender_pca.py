import os
import os.path as osp
import numpy as np
from global_var import ROOT


pca_range = {
    't-shirt': {
        'neutral': [[-0.4, 0, 0, 0], [2, 1.5, 0, 0], -0.004],
        'male': [[0.6, 0, 0, 0], [2, 1.5, 0, 0], 0.008],
        'female': [[-1, 0, 0, 0], [2, 1.5, 0, 0], -0.016]
    },
    # 'shirt': {
    #     'neutral': [[0.5, 0, 0, 0], [1.5, 2., 0, 0]],
    #     'male': [[2, 0, 0, 0], [1.5, 2, 0, 0], 0.004],
    #     'female': [[-0.4, 0, 0, 0], [1.5, 2, 0, 0], -0.008  ]
    # },
    'shirt': {
        'neutral': [[0., 0, 0, 0], [1, 1.5, 0, 0]],
        'male': [[1.5, 0, 0, 0], [1, 1.5, 0, 0]],
        'female': [[-0.5, 0, 0, 0], [1, 1.5, 0, 0]]
    },
    'pant': {
        'neutral': [[0.5, 0.5, 0, 0], [1, 0.5, 0, 0], -0.014],
        'male': [[1, 0.5, 0, 0], [1, 0.5, 0, 0], -0.03],
        'female': [[0, 0.5, 0, 0], [1, 0.5, 0, 0], 0.01]
    },
    'skirt_orig': {
        'female': [[-2, 0, 0, 0], [2, 0, 0, 0], 0]
    },
    'skirt': {
        'female': [[-3, -0.6, 0, 0], [3, 0.6, 0, 0], 0]
    },
    'short-pant': {
        'female': [[0, -0.6, 0, 0], [3, 1, 0, 0]],
        'male': [[-1, 0, 0, 0], [2, 1, 0, 0]],
    }

}


if __name__ == '__main__':
    for gc in ['skirt']:
        orig_pca_path = osp.join(ROOT, 'pca', 'style_model_{}.npz'.format(gc))
        orig_pca = np.load(orig_pca_path)
        for gender in ['female']:
            pca_r = pca_range[gc][gender]
            save_dir = osp.join(ROOT, '{}_{}'.format(gc, gender))
            os.makedirs(save_dir, exist_ok=True)
            np.savez(osp.join(save_dir, 'style_model.npz'),
                     pca_w=orig_pca['pca_w'], mean=orig_pca['mean'],
                     coeff_mean=np.array(pca_r[0], dtype=np.float32),
                     coeff_range=np.array(pca_r[1], dtype=np.float32))