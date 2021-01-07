import numpy as np
import chumpy as ch
from psbody.mesh import Mesh


def delete_verts(faces, delete_mat):
    new_faces = []
    for face in faces:
        to_delete = False
        for fidx in face:
            if delete_mat[fidx]:
                to_delete = True
                break
        if not to_delete:
            new_faces.append(face)
    if len(new_faces) > 0:
        new_faces = np.stack(new_faces, 0)
    else:
        new_faces = None
    return new_faces


def divide_obj(obj_path):
    with open(obj_path) as f:
        lines = f.read().splitlines()
    items = {'v': [], 'vn': [], 'f': []}
    last_type = ''
    for line in lines:
        line_type = line.split(' ')[0]
        if line_type in items:
            if line_type == 'v' or line_type == 'vn':
                if line_type != last_type:
                    items[line_type].append([line+'\n'])
                else:
                    items[line_type][-1].append(line+'\n')
            if line_type == 'f':
                single_face = line.split(' ')[1:]
                for fi, sf in enumerate(single_face):
                    single_face[fi] = sf.split('//')
                single_face = np.array(single_face).astype(np.int)
                if line_type != last_type:
                    items['f'].append([single_face])
                else:
                    items['f'][-1].append(single_face)
            last_type = line_type
    v_num, vn_num = 0, 0
    for obj_idx, one_obj_faces in enumerate(items['f']):
        if obj_idx >= 1:
            v_num += len(items['v'][obj_idx-1])
            vn_num += len(items['vn'][obj_idx-1])
        for fi, one_obj_face in enumerate(one_obj_faces):
            one_obj_face[:, 0] -= v_num
            one_obj_face[:, 1] -= vn_num
            new_line = 'f'
            for face_v in one_obj_face:
                new_line += " {}//{}".format(face_v[0], face_v[1])
            new_line += '\n'
            one_obj_faces[fi] = new_line
    return items


def unpose_garment(smpl, v_free, vert_indices=None):
    smpl.v_personal[:] = 0
    c = smpl[vert_indices]
    E = {
        'v_personal_high': c - v_free
    }
    ch.minimize(E, x0=[smpl.v_personal], options={'e_3': .00001})
    smpl.pose[:] = 0
    smpl.trans[:] = 0

    return Mesh(smpl.r, smpl.f).keep_vertices(vert_indices), np.array(smpl.v_personal)


def unpose_skirt(smpl, verts):
    rotmat =smpl.A.r[:, :, 0]
    # inv_rotmat = np.linalg.inv(rotmat)
    verts_homo = np.hstack((verts, np.ones((verts.shape[0], 1))))
    verts = verts_homo.dot(np.linalg.inv(rotmat.T))[:, :3]
    return verts


def get_selfintersections(v, f):
    import pymesh
    from pymesh.selfintersection import detect_self_intersection
    import numpy as np
    mspy = pymesh.form_mesh(v, f)
    face_pairs = detect_self_intersection(mspy)
    mspy.add_attribute('face_area')
    face_areas = mspy.get_attribute('face_area')
    intersecting_area = face_areas[np.unique(face_pairs.ravel())].sum()
    return face_pairs, intersecting_area


if __name__ == '__main__':
    import global_var
    import os
    items = divide_obj(os.path.join(global_var.ROOT, 'static_pose', 'rp_aaron_posed_006_30k', 'result_7.obj'))
    obj_num = len(items['f'])
    for obj_idx in range(obj_num):
        with open("{}.obj".format(obj_idx), 'w') as f:
            f.writelines(items['v'][obj_idx])
            f.writelines(items['vn'][obj_idx])
            f.writelines(items['f'][obj_idx])