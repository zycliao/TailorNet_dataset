import os
from global_var import ROOT

if __name__ == '__main__':
    gender = 'female'
    gc = 'skirt'
    data_dir = os.path.join(ROOT, '{}_{}'.format(gc, gender))
    bad_dir = os.path.join(data_dir, 'style_shape_vis', 'bad')
    ss_dir = os.path.join(data_dir, 'style_shape')

    all_ss = [k.replace('.obj', '') for k in os.listdir(ss_dir) if k.endswith('.obj') and k.startswith('beta')]
    all_bad = [k.replace('.jpg', '') for k in os.listdir(bad_dir)]

    with open(os.path.join(data_dir, 'avail.txt'), 'w') as f:
        for ss in all_ss:
            if ss in all_bad:
                continue
            ss_item = ss.replace('beta', '').replace('gamma', '')
            f.write(ss_item+'\n')
