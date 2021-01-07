import os
import os.path as osp
import numpy as np
from global_var import ROOT


if __name__ == '__main__':
    all_garment_class = ['t-shirt', 'shirt', 'pant', 'short-pant', 'skirt']
    genders = ['female', 'male']
    splits = ['train', 'test']
    img_size = 512

    split_idx = np.load(osp.join(ROOT, 'split_static_pose_shape.npz'))
    train_idx = split_idx['train']
    test_idx = split_idx['test']

    print("—" * 78)
    print("|{:>10}|{:>10}|{:>21}|{:>21}|{:>10}|".format('', '', 'train style_shape', 'test style_shape', ''))
    print("|{:>10}|".format('class'), end='')
    print("{:>10}|".format('gender'), end='')
    print("{:>10}|{:>10}|".format('train pose', 'test pose'), end='')
    print("{:>10}|{:>10}|".format('train pose', 'test pose'), end='')
    print("{:>10}|".format('total'))
    print("—" * 78)

    count_split = np.array([0, 0, 0, 0])
    for garment_class in all_garment_class:
        for gender in genders:
            if garment_class == 'skirt' and gender == 'male':
                continue
            print("|{:>10}|".format(garment_class), end='')
            print("{:>10}|".format(gender), end='')
            count_one_class = 0
            for split in splits:
                pose_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'pose')
                if split == 'train':
                    ss_path = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'pivots.txt')
                else:
                    ss_path = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'test.txt')
                with open(ss_path) as f:
                    all_ss = f.read().strip().splitlines()
                train_num = 0
                test_num = 0
                for ss in all_ss:
                    pose_ss_dir = osp.join(pose_dir, ss)
                    if not osp.exists(pose_ss_dir):
                        continue
                    unpose_names = [k for k in os.listdir(pose_ss_dir) if k.startswith('unposed') and k.endswith('.npy')]
                    for unpose_name in unpose_names:
                        seq_str = unpose_name.replace('unposed_', '').replace('.npy', '')
                        pose_path = osp.join(pose_ss_dir, 'poses_{}.npz'.format(seq_str))

                        pose_idx = np.load(pose_path)['pose_order']
                        train_num += np.sum(np.in1d(pose_idx, train_idx))
                        test_num += np.sum(np.in1d(pose_idx, test_idx))
                print("{:>10}|".format(train_num), end='')
                print("{:>10}|".format(test_num), end='')
                count_one_class += train_num
                count_one_class += test_num
                if split == 'train':
                    count_split[0] += train_num
                    count_split[1] += test_num
                else:
                    count_split[2] += train_num
                    count_split[3] += test_num
            print("{:>10}|".format(count_one_class))
    print("|{:>21}|".format('total'), end='')
    for n in count_split:
        print("{:>10}|".format(n), end='')
    print("{:>10}|".format(np.sum(count_split)))
    print("—" * 78)


