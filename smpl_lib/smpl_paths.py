import numpy as np
# import cPickle as pkl
import pickle as pkl
from smpl_lib.serialization import backwards_compatibility_replacements, load_model
import scipy.sparse as sp
from global_var import SMPL_PATH_MALE, SMPL_PATH_FEMALE


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vert_opposites_per_edge(mesh_v, mesh_f):
    """Returns a dictionary from vertidx-pairs to opposites.
    For example, a key consist of [4,5)] meaning the edge between
    vertices 4 and 5, and a value might be [10,11] which are the indices
    of the vertices opposing this edge."""
    result = {}
    for f in mesh_f:
        for i in range(3):
            key = [f[i], f[(i+1)%3]]
            key.sort()
            key = tuple(key)
            val = f[(i+2)%3]

            if key in result:
                result[key].append(val)
            else:
                result[key] = [val]
    return result

def get_vertices_per_edge(mesh_v, mesh_f):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()"""

    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:,0] < result[:,1]] # for uniqueness

    return result

def loop_subdivider(mesh_v, mesh_f):

    IS = []
    JS = []
    data = []

    vc = get_vert_connectivity(mesh_v, mesh_f)
    ve = get_vertices_per_edge(mesh_v, mesh_f)
    vo = get_vert_opposites_per_edge(mesh_v, mesh_f)

    if True:
        # New values for each vertex
        for idx in range(len(mesh_v)):

            # find neighboring vertices
            nbrs = np.nonzero(vc[:,idx])[0]

            nn = len(nbrs)

            if nn < 3:
                wt = 0.
            elif nn == 3:
                wt = 3./16.
            elif nn > 3:
                wt = 3. / (8. * nn)
            else:
                raise Exception('nn should be 3 or more')
            if wt > 0.:
                for nbr in nbrs:
                    IS.append(idx)
                    JS.append(nbr)
                    data.append(wt)

            JS.append(idx)
            IS.append(idx)
            data.append(1. - (wt * nn))

    start = len(mesh_v)
    edge_to_midpoint = {}

    if True:
        # New values for each edge:
        # new edge verts depend on the verts they span
        for idx, vs in enumerate(ve):

            vsl = list(vs)
            vsl.sort()
            IS.append(start + idx)
            IS.append(start + idx)
            JS.append(vsl[0])
            JS.append(vsl[1])
            data.append(3./8)
            data.append(3./8)

            opposites = vo[(vsl[0], vsl[1])]
            for opp in opposites:
                IS.append(start + idx)
                JS.append(opp)
                data.append(2./8./len(opposites))

            edge_to_midpoint[(vsl[0], vsl[1])] = start + idx
            edge_to_midpoint[(vsl[1], vsl[0])] = start + idx

    f = []

    for f_i, old_f in enumerate(mesh_f):
        ff = np.concatenate((old_f, old_f))

        for i in range(3):
            v0 = edge_to_midpoint[(ff[i], ff[i+1])]
            v1 = ff[i+1]
            v2 = edge_to_midpoint[(ff[i+1], ff[i+2])]
            f.append(row(np.array([v0,v1,v2])))
        v0 = edge_to_midpoint[(ff[0], ff[1])]
        v1 = edge_to_midpoint[(ff[1], ff[2])]
        v2 = edge_to_midpoint[(ff[2], ff[3])]
        f.append(row(np.array([v0,v1,v2])))

    f = np.vstack(f)

    IS = np.array(IS, dtype=np.uint32)
    JS = np.array(JS, dtype=np.uint32)

    if True: # for x,y,z coords
        IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
        JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
        data = np.concatenate ((data,data,data))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij))

    return mtx, f

def get_hres(v, f):
    """
    Get an upsampled version of the mesh.
    OUTPUT:
        - nv: new vertices
        - nf: faces of the upsampled
        - mapping: mapping from low res to high res
    """
    (mapping, nf) = loop_subdivider(v, f)
    nv = mapping.dot(v.ravel()).reshape(-1, 3)
    return (nv, nf, mapping)


# smpl_vt_ft_path = '/BS/bharat/work/MGN_final_release/assets/smpl_vt_ft.pkl'
class SmplPaths:
    def __init__(self, project_dir='', exp_name='', gender='neutral', garment=''):
        self.project_dir = project_dir
        # experiments name
        self.exp_name = exp_name
        self.gender = gender
        self.garment = garment

    def get_smpl_file(self):
        if self.gender == 'male':
            return SMPL_PATH_MALE
        elif self.gender == 'female':
            return SMPL_PATH_FEMALE
        else:
            print("Get proper neutral model and check for neutral betas.")
            raise NotImplemented

    def get_smpl(self):
        smpl_m = load_model(self.get_smpl_file())
        smpl_m.gender = self.gender
        return smpl_m

    def get_hres_smpl_model_data(self):

        dd = pkl.load(open(self.get_smpl_file(), 'rb'), encoding='latin1')
        backwards_compatibility_replacements(dd)

        hv, hf, mapping = get_hres(dd['v_template'], dd['f'])

        num_betas = dd['shapedirs'].shape[-1]
        J_reg = dd['J_regressor'].asformat('csr')

        model = {
            'v_template': hv,
            'weights': np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]),
            'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207),
            'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
            'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
            'kintree_table': dd['kintree_table'],
            'bs_type': dd['bs_type'],
            'bs_style': dd['bs_style'],
            'J': dd['J'],
            'f': hf,
        }

        return model

    def get_hres_smpl(self):
        smpl_m = load_model(self.get_hres_smpl_model_data())
        smpl_m.gender = self.gender
        return smpl_m

    # @staticmethod
    # def get_vt_ft():
    #     vt, ft = pkl.load(open(smpl_vt_ft_path))
    #     return vt, ft
    #
    # @staticmethod
    # def get_vt_ft_hres():
    #     vt, ft = SmplPaths.get_vt_ft()
    #     vt, ft, _ = get_hres(np.hstack((vt, np.ones((vt.shape[0], 1)))), ft)
    #     return vt[:, :2], ft


if __name__ == '__main__':
    dp = SmplPaths(gender='female')
    smpl_file = dp.get_smpl_file()
    smpl = dp.get_smpl()
    smpl_hres = dp.get_hres_smpl()
    print(smpl_file)