import numpy as np
import scipy
import scipy.sparse as sp
from psbody.mesh import Mesh
import stat


def get_edges2face(faces):
    from itertools import combinations
    from collections import OrderedDict
    # Returns a structure that contains the faces corresponding to every edge
    edges = OrderedDict()
    for iface, f in enumerate(faces):
        sorted_face_edges = tuple(combinations(sorted(f), 2))
        for sorted_face_edge in sorted_face_edges:
            if sorted_face_edge in edges:
                edges[sorted_face_edge].faces.add(iface)
            else:
                edges[sorted_face_edge] = lambda:0
                edges[sorted_face_edge].faces = set([iface])
    return edges


def get_boundary_verts(verts, faces, connected_boundaries=True, connected_faces=False):
    """
     Given a mesh returns boundary vertices
     if connected_boundaries is True it returs a list of lists
     OUTPUT:
        boundary_verts: list of verts
        cnct_bound_verts: list of list containing the N ordered rings of the mesh
    """
    MIN_NUM_VERTS_RING = 10
    # Ordred dictionary
    edge_dict = get_edges2face(faces)
    boundary_verts = []
    boundary_edges = []
    boundary_faces = []
    for edge, (key, val) in enumerate(edge_dict.items()):
        if len(val.faces) == 1:
            boundary_verts += list(key)
            boundary_edges.append(edge)
            for face_id in val.faces:
                boundary_faces.append(face_id)
    boundary_verts = list(set(boundary_verts))
    if not connected_boundaries:
        return boundary_verts
    n_removed_verts = 0
    if connected_boundaries:
        edge_mat = np.array(list(edge_dict.keys()))
        # Edges on the boundary
        edge_mat = edge_mat[np.array(boundary_edges)]

        # check that every vertex is shared by only two edges
        for v in boundary_verts:
            if np.sum(edge_mat == v) != 2:
                import ipdb; ipdb.set_trace();
                raise ValueError('The boundary edges are not closed loops!')

        cnct_bound_verts = []
        while len(edge_mat > 0):
            # boundary verts, indices of conected boundary verts in order
            bverts = []
            orig_vert = edge_mat[0, 0]
            bverts.append(orig_vert)
            vert = edge_mat[0, 1]
            edge = 0
            while orig_vert != vert:
                bverts.append(vert)
                # remove edge from queue
                edge_mask = np.ones(edge_mat.shape[0], dtype=bool)
                edge_mask[edge] = False
                edge_mat = edge_mat[edge_mask]
                edge = np.where(np.sum(edge_mat == vert, axis=1) > 0)[0]
                tmp = edge_mat[edge]
                vert = tmp[tmp != vert][0]
            # remove the last edge
            edge_mask = np.ones(edge_mat.shape[0], dtype=bool)
            edge_mask[edge] = False
            edge_mat = edge_mat[edge_mask]
            if len(bverts) > MIN_NUM_VERTS_RING:
                # add ring to the list
                cnct_bound_verts.append(bverts)
            else:
                n_removed_verts += len(bverts)
    count = 0
    for ring in cnct_bound_verts: count += len(ring)
    assert(len(boundary_verts) - n_removed_verts == count), "Error computing boundary rings !!"

    if connected_faces:
        return (boundary_verts, boundary_faces, cnct_bound_verts)
    else:
        return (boundary_verts, cnct_bound_verts)


def numpy_laplacian_uniform(v, f):
    """ Compute laplacian operator on part_mesh. This can be cached.
    """
    import scipy.sparse as sp
    from sklearn.preprocessing import normalize
    from psbody.mesh.topology.connectivity import get_vert_connectivity

    connectivity = get_vert_connectivity(Mesh(v=v, f=f))
    # connectivity is a sparse matrix, and np.clip can not applied directly on
    # a sparse matrix.
    connectivity.data = np.clip(connectivity.data, 0, 1)
    lap = normalize(connectivity, norm='l1', axis=1)
    lap = lap - sp.eye(connectivity.shape[0])

    return lap


def numpy_laplacian_cot(v, f):
    n = len(v)

    v_a = f[:, 0]
    v_b = f[:, 1]
    v_c = f[:, 2]

    ab = v[v_a] - v[v_b]
    bc = v[v_b] - v[v_c]
    ca = v[v_c] - v[v_a]

    cot_a = -1 * (ab * ca).sum(axis=1) / (np.sqrt(np.sum(np.cross(ab, ca) ** 2, axis=-1)) + 1.e-10)
    cot_b = -1 * (bc * ab).sum(axis=1) / (np.sqrt(np.sum(np.cross(bc, ab) ** 2, axis=-1)) + 1.e-10)
    cot_c = -1 * (ca * bc).sum(axis=1) / (np.sqrt(np.sum(np.cross(ca, bc) ** 2, axis=-1)) + 1.e-10)

    I = np.concatenate((v_a, v_c, v_a, v_b, v_b, v_c))
    J = np.concatenate((v_c, v_a, v_b, v_a, v_c, v_b))
    W = 0.5 * np.concatenate((cot_b, cot_b, cot_c, cot_c, cot_a, cot_a))

    L = sp.csr_matrix((W, (I, J)), shape=(n, n))
    L = L - sp.spdiags(L * np.ones(n), 0, n, n)

    return L


class DiffusionSmoothing(object):

    def __init__(self, v, f, Ltype="cotangent"):

        assert(Ltype in ["cotangent", "uniform"])
        self.Ltype = Ltype
        self.num_v = v.shape[0]
        self.f = f
        self.set_boundary_ids_and_mats(v, f)
        self.L = None if self.Ltype == "cotangent" else self.get_uniform_lap_smoothing(v)

    def get_uniform_lap_smoothing(self, v):
        L = numpy_laplacian_uniform(v, self.f)

        # remove rows corresponding to boundary vertices
        for row in self.b_ids:
            L.data[L.indptr[row]:L.indptr[row + 1]] = 0
        L.eliminate_zeros()

        num_b = self.b_ids.shape[0]
        I = np.tile(self.b_ids, 3)
        J = np.hstack((
            self.b_ids,
            self.b_ids[self.l_ids],
            self.b_ids[self.r_ids],
        ))
        W = np.hstack((
            -1 * np.ones(num_b),
            0.5 * np.ones(num_b),
            0.5 * np.ones(num_b),
        ))
        mat = sp.csr_matrix((W, (I, J)), shape=(self.num_v, self.num_v))
        L = L + mat
        return L

    def set_boundary_ids_and_mats(self, v, f):
        _, b_rings = get_boundary_verts(v, f)

        def shift_left(ls, k):
            return ls[k:] + ls[:k]

        b_ids = []
        l_ids = []
        r_ids = []
        for rg in b_rings:
            tmp = list(range(len(b_ids), len(b_ids) + len(rg)))
            ltmp = shift_left(tmp, 1)
            rtmp = shift_left(tmp, -1)
            l_ids.extend(ltmp)
            r_ids.extend(rtmp)

            b_ids.extend(rg)

        b_ids = np.asarray(b_ids)
        num_b = b_ids.shape[0]
        m_ids = np.arange(num_b)
        l_ids = np.asarray(l_ids)
        r_ids = np.asarray(r_ids)

        self.right_edge_mat = sp.csr_matrix((
            np.hstack((-1*np.ones(num_b), np.ones(num_b))),
            (np.hstack((m_ids, m_ids)), np.hstack((m_ids, r_ids)))
        ), shape=(num_b, num_b)
        )

        self.left_edge_mat = sp.csr_matrix((
            np.hstack((-1 * np.ones(num_b), np.ones(num_b))),
            (np.hstack((m_ids, m_ids)), np.hstack((m_ids, l_ids)))
        ), shape=(num_b, num_b)
        )

        self.b_ids = b_ids
        self.l_ids = l_ids
        self.r_ids = r_ids

    def smooth_cotlap(self, verts, smoothness=0.03):
        L = numpy_laplacian_cot(verts, self.f)
        new_verts = verts + smoothness * L.dot(verts)

        b_verts = verts[self.b_ids]
        le = 1. / (np.linalg.norm(self.left_edge_mat.dot(b_verts), axis=-1) + 1.0e-10)
        ri = 1. / (np.linalg.norm(self.right_edge_mat.dot(b_verts), axis=-1) + 1.0e-10)

        num_b = b_verts.shape[0]
        I = np.tile(np.arange(num_b), 3)
        J = np.hstack((
            np.arange(num_b),
            self.l_ids,
            self.r_ids,
        ))
        W = np.hstack((
            -1*np.ones(num_b),
            le / (le + ri),
            ri / (le + ri),
        ))
        mat = sp.csr_matrix((W, (I, J)), shape=(num_b, num_b))
        new_verts[self.b_ids] = verts[self.b_ids] + smoothness * mat.dot(verts[self.b_ids])
        return new_verts

    def smooth_uniform(self, verts, smoothness=0.03):
        new_verts = verts + smoothness * self.L.dot(verts)
        return new_verts

    def smooth(self, verts, smoothness=0.03):
        return self.smooth_uniform(verts, smoothness) if self.Ltype == "uniform" else self.smooth_cotlap(verts, smoothness)


if __name__ == "__main__":
    import os
    import global_var
    from utils.args import get_args
    from tqdm import tqdm

    garment_class = 'smooth_TShirtNoCoat'
    gender, list_name = get_args()

    with open(os.path.join(global_var.ROOT, '{}_{}.txt').format(gender, list_name)) as f:
        avail_items = f.read().splitlines()
    avail_items = [k.split('\t') for k in avail_items]
    people_names = [k[0] for k in avail_items if k[1] == garment_class]
    shape_root = os.path.join(global_var.ROOT, 'neutral_shape_static_pose_new')
    smoothing = None

    shape_names = ["{:02d}".format(k) for k in range(0, 100)]

    for people_name, garment_class in tqdm(avail_items):
        shape_static_pose_people = os.path.join(shape_root, people_name)
        for shape_name in shape_names:
            garment_path = os.path.join(shape_static_pose_people, '{}_{}.obj'.format(shape_name, garment_class))
            if not os.path.exists(garment_path):
                print("{} doesn't exist".format(garment_path))
            try:
                m = Mesh(filename=garment_path)
            except AttributeError as e:
                print(e)
                print(garment_path)
                exit()

            if smoothing is None:
                smoothing = DiffusionSmoothing(m.v, m.f, Ltype="cotangent")
            steps = [30]

            verts_smooth = m.v.copy()
            for i, step in enumerate(steps):
                smooth_name = '_sm{}'.format(step)
                dst_path = os.path.join(shape_static_pose_people, '{}{}_{}.obj'.format(shape_name, smooth_name, garment_class))
                if os.path.exists(dst_path):
                    print("{} exists. Skip".format(dst_path))
                for _ in range(step):
                    verts_smooth = smoothing.smooth(verts_smooth, smoothness=0.03)
                ms_smooth = Mesh(v=verts_smooth, f=m.f)

                ms_smooth.write_obj(dst_path)
                os.chmod(dst_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)