import os
import numpy as np
from global_var import ROOT


def read_ss(path):
    with open(path) as f:
        c = f.read().strip().splitlines()
    return c

if __name__ == '__main__':
    garment_class = 'skirt'
    gender = 'female'
    test_num = 20

    data_dir = os.path.join(ROOT, f'{garment_class}_{gender}')
    pivots = read_ss(os.path.join(data_dir, 'pivots.txt'))
    all_ss = read_ss(os.path.join(data_dir, 'avail.txt'))
    other_ss = []
    for ss in all_ss:
        if ss not in pivots:
            other_ss.append(ss)
    ind = np.arange(len(other_ss))
    np.random.shuffle(ind)
    ind = ind[:test_num]
    other_ss = np.array(other_ss)
    test_ss = other_ss[ind]
    with open(os.path.join(data_dir, 'test.txt'), 'w') as f:
        for ss in test_ss:
            f.write(ss)
            f.write('\n')