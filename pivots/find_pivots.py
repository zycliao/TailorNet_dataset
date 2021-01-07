import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from global_var import ROOT


def calc_dist(v1, v2):
    if isinstance(v1, np.ndarray):
        return np.mean(np.sqrt(np.sum(np.square(v1 - v2), -1)))
    elif isinstance(v1, torch.Tensor):
        return torch.mean(torch.sqrt(torch.sum(torch.pow(v1 - v2, 2), -1)))


class Comb(nn.Module):
    def __init__(self, nbasis):
        super(Comb, self).__init__()
        self.nbasis = nbasis
        self.softmax = nn.Softmax()
        self.w = nn.Parameter(torch.zeros(nbasis), requires_grad=True)

    def forward(self, x):
        # x: (N, v, 3)
        norm_w = self.softmax(self.w)
        return torch.einsum('n,nvt->vt', norm_w, x)

    def reset(self):
        self.w.data *= 0

    @property
    def norm_w(self):
        return self.softmax(self.w.data.detach())


class WOptimizer(object):
    """
    Use this class to optimize the coefficients of a linear combination
    """
    def __init__(self, basis, lr=1e-1, max_iter=100):
        if isinstance(basis, list) or isinstance(basis, tuple):
            basis = torch.stack(basis, 0)
        self.basis = basis
        self.nbasis = basis.shape[0]
        self.comb = Comb(self.nbasis)
        self.lr = lr
        self.max_iter = max_iter
        self.criterons = torch.nn.L1Loss()

    def __call__(self, x):
        self.comb.reset()
        x = x.detach()
        optimizer = torch.optim.Adam(self.comb.parameters(), self.lr)
        dist = np.inf
        comb_result = None
        for i in range(self.max_iter):
            optimizer.zero_grad()
            comb_result = self.comb(self.basis)
            loss = self.criterons(comb_result, x)
            loss.backward()
            optimizer.step()
            d = calc_dist(comb_result, x).detach().cpu().item()
            if d < dist:
                dist = d
        return comb_result.detach(), self.comb.norm_w, dist

    def cuda(self):
        self.comb.cuda()

class Runner(object):
    def __init__(self, garment_class, gender, num_pivots=20):
        self.garment_class = garment_class
        self.gender = gender
        self.K = num_pivots
        self.lr = 1e-1
        self.max_iter = 100
        self.criterons = torch.nn.L1Loss()

        self.data_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender))
        self.ss_dir = osp.join(self.data_dir, 'style_shape')

        self.verts_dict = {}
        with open(osp.join(self.data_dir, 'avail.txt')) as f:
            all_ss = f.read().strip().splitlines()
        for ss in all_ss:
            beta_str, gamma_str = ss.split('_')
            d = np.load(osp.join(self.ss_dir, 'beta{}_gamma{}.npy'.format(beta_str, gamma_str)))
            self.verts_dict["{}_{}".format(beta_str, gamma_str)] = torch.from_numpy(d.astype(np.float32))

        self.current_basis = []

    def iter(self):
        nbasis = len(self.current_basis)
        basis = []
        for bname in self.current_basis:
            basis.append(self.verts_dict[bname])
        basis = torch.stack(basis, 0)
        woptim = WOptimizer(basis)

        names, dists = [], []
        for name, verts in tqdm(self.verts_dict.items()):
            comb_results, _, dist = woptim(verts)
            # print("{} dist: {:.4f} mm".format(name, dist*1000))
            names.append(name)
            dists.append(dist)
        dists = np.array(dists)
        mean_dist = np.mean(dists)
        max_i = np.argmax(dists)
        print("basis num: {}".format(nbasis))
        print("max dist: {} mm".format(dists[max_i] * 1000.))
        print("mean dist: {} mm".format(mean_dist * 1000.))
        self.current_basis.append(names[int(max_i)])

    def baseline(self):
        import random
        all_names = list(self.verts_dict.keys())
        random.shuffle(all_names)
        basis = []
        for name in all_names[:self.K]:
            basis.append(self.verts_dict[name])
        basis = torch.stack(basis, 0)
        nbasis = basis.shape[0]
        woptim = WOptimizer(basis)

        names, dists = [], []
        for name, verts in tqdm(self.verts_dict.items()):
            comb_results, _, dist = woptim(verts)
            # print("{} dist: {:.4f} mm".format(name, dist*1000))
            names.append(name)
            dists.append(dist)
        dists = np.array(dists)
        mean_dist = np.mean(dists)
        max_i = np.argmax(dists)
        print("Baseline---basis num: {}".format(nbasis))
        print("max dist: {} mm".format(dists[max_i] * 1000.))
        print("mean dist: {} mm".format(mean_dist * 1000.))

    def baseline_kmeans(self, k=None):
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.K if k is None else k)
        all_verts = []
        all_names = []
        for name, verts in self.verts_dict.items():
            all_verts.append(verts.numpy())
            all_names.append(name)
        all_verts = np.stack(all_verts, 0).reshape(len(all_verts), -1)
        kmeans.fit(all_verts)
        centers = kmeans.cluster_centers_
        basis = []
        basis_names = []
        for i in range(len(centers)):
            center = centers[i: i+1]
            dist = np.mean(np.square(all_verts - center), -1)
            min_idx = np.argmin(dist)
            one_basis = all_verts[min_idx].reshape([-1, 3])
            basis.append(one_basis)
            basis_names.append(all_names[min_idx])
        basis = torch.from_numpy(np.array(basis))

        woptim = WOptimizer(basis)

        names, dists = [], []
        for name, verts in tqdm(self.verts_dict.items()):
            comb_results, _, dist = woptim(verts)
            # print("{} dist: {:.4f} mm".format(name, dist*1000))
            names.append(name)
            dists.append(dist)
        dists = np.array(dists)
        mean_dist = np.mean(dists)
        max_i = np.argmax(dists)
        print("KMeans---basis num: {}".format(k))
        print("max dist: {} mm".format(dists[max_i] * 1000.))
        print("mean dist: {} mm".format(mean_dist * 1000.))
        return basis_names


    def save_basis(self):
        with open(os.path.join(self.data_dir, 'pivots.txt'), 'w') as f:
            for bname in self.current_basis:
                f.write(bname+'\n')



if __name__ == '__main__':
    garment_class = 'skirt'
    gender = 'female'
    num_pivots = 20
    num_kmeans = 10
    runner = Runner(garment_class, gender, num_pivots)
    # test kmeans
    # runner.baseline_kmeans()

    # kmeans + greedy
    runner.current_basis.extend(runner.baseline_kmeans(num_kmeans))
    left_k = runner.K - len(runner.current_basis)
    for _ in range(left_k):
        runner.iter()
    runner.save_basis()
    runner.iter()
