"""
Interactively visualize style PCA
"""
import os
import os.path as osp
import cv2
import pickle
import numpy as np
import trimesh
from utils.renderer import Renderer
from sklearn.decomposition import PCA
import global_var


class Controller(object):
    def __init__(self, gc, gender):
        data_dir = osp.join(global_var.ROOT, 'pca')
        style_model = np.load(osp.join(data_dir, 'style_model_{}.npz'.format(gc)))

        self.renderer = Renderer(512)
        self.img = None
        self.body = trimesh.load(osp.join(global_var.ROOT, 'smpl', 'hres_{}.obj'.format(gender)), process=False)

        with open(osp.join(global_var.ROOT, 'garment_class_info.pkl'), 'rb') as f:
            garment_meta = pickle.load(f)
        # self.vert_indcies = garment_meta[gc]['vert_indices']
        self.f = garment_meta[gc]['f']

        self.gamma = np.zeros([1, 4], dtype=np.float32)
        self.pca = PCA(n_components=4)
        self.pca.components_ = style_model['pca_w']
        self.pca.mean_ = style_model['mean']
        win_name = '{}_{}'.format(gc, gender)
        cv2.namedWindow(win_name)
        cv2.createTrackbar('0', win_name, 100, 200, self.value_change)
        cv2.createTrackbar('1', win_name, 100, 200, self.value_change)
        cv2.createTrackbar('2', win_name, 100, 200, self.value_change)
        cv2.createTrackbar('3', win_name, 100, 200, self.value_change)
        cv2.createTrackbar('trans', win_name, 100, 200, self.value_change)
        self.trans = 0
        self.win_name = win_name

        self.render()

    def value_change(self, x):
        g0 = cv2.getTrackbarPos('0', self.win_name)
        g1 = cv2.getTrackbarPos('1', self.win_name)
        g2 = cv2.getTrackbarPos('2', self.win_name)
        g3 = cv2.getTrackbarPos('3', self.win_name)
        trans = cv2.getTrackbarPos('trans', self.win_name)
        self.gamma[0] = [self.convert(g0), self.convert(g1), self.convert(g2), self.convert(g3)]
        self.trans = (trans - 100) / 1000.
        self.render()
        print(f"{self.gamma[0]}\t{self.trans*100}")

    def convert(self, v):
        return (v - 100.) / 20.

    def render(self):
        v = self.pca.inverse_transform(self.gamma).reshape([-1, 3])
        v[:, 1] += self.trans
        self.img = self.renderer([v, self.body.vertices], [self.f, self.body.faces],
            [np.array([0.8, 0.5, 0.3]), np.array([0.6, 0.6, 0.9])], trans=(1, 0, 0.3))


if __name__ == '__main__':
    gender = 'female'
    garment_class = 'skirt'
    controller = Controller(garment_class, gender)
    while True:
        cv2.imshow(controller.win_name, controller.img)
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
    cv2.destroyAllWindows()
