import os
from os.path import join as opj
import shutil
from datetime import datetime
import numpy as np
from plyfile import PlyData
import struct
import global_var


def  read_ply(fname):
    plydata = PlyData.read(fname)
    vert_ele = plydata['vertex']
    verts = np.stack([vert_ele['x'], vert_ele['y'], vert_ele['z']], 1)
    faces = np.stack(list(plydata['face']['vertex_indices']), 0)
    return verts, faces


def read_obj(path):
    """
    read verts and faces from obj file. This func will convert quad mesh to triangle mesh
    """
    with open(path) as f:
        lines = f.read().splitlines()
    verts = []
    faces = []
    for line in lines:
        if line.startswith('v '):
            verts.append(np.array([float(k) for k in line.split(' ')[1:]]))
        elif line.startswith('f '):
            try:
                onef = np.array([int(k) for k in line.split(' ')[1:]])
            except ValueError:
                continue
            if len(onef) == 4:
                faces.append(onef[[0, 1, 2]])
                faces.append(onef[[0, 2, 3]])
            elif len(onef) > 4:
                pass
            else:
                faces.append(onef)
    if len(faces) == 0:
        return np.stack(verts), None
    else:
        return np.stack(verts), np.stack(faces)-1


def write_obj(verts, faces, path, color_idx=None):
    faces = faces + 1
    with open(path, 'w') as f:
        for vidx, v in enumerate(verts):
            if color_idx is not None and color_idx[vidx]:
                f.write("v {:.5f} {:.5f} {:.5f} 1 0 0\n".format(v[0], v[1], v[2]))
            else:
                f.write("v {:.5f} {:.5f} {:.5f}\n".format(v[0], v[1], v[2]))
        for fa in faces:
            f.write("f {:d} {:d} {:d}\n".format(fa[0], fa[1], fa[2]))


def save_pc2(vertices, path):
    # vertices: (N, V, 3), N is the number of frames, V is the number of vertices
    # path: a .pc2 file
    nframes, nverts, _ = vertices.shape
    with open(path, 'wb') as f:
        headerStr = struct.pack('<12siiffi', b'POINTCACHE2\0',
                                1, nverts, 1, 1, nframes)
        f.write(headerStr)
        v = vertices.reshape(-1, 3).astype(np.float32)
        for v_ in v:
            f.write(struct.pack('<fff', v_[0], v_[1], v_[2]))


def read_pc2(path):
    with open(path, 'rb') as f:
        head_fmt = '<12siiffi'
        data_fmt = '<fff'
        head_unpack = struct.Struct(head_fmt).unpack_from
        data_unpack = struct.Struct(data_fmt).unpack_from
        data_size = struct.calcsize(data_fmt)
        headerStr = f.read(struct.calcsize(head_fmt))
        head = head_unpack(headerStr)
        nverts, nframes = head[2], head[5]
        data = []
        for i in range(nverts*nframes):
            data_line = f.read(data_size)
            if len(data_line) != data_size:
                return None
            data.append(list(data_unpack(data_line)))
        data = np.array(data).reshape([nframes, nverts, 3])
    return data


def backup_file(src, dst):
    # if a directory is not a package nor the root directory. skip it.
    if not os.path.exists(opj(src, '__init__.py')) and src != global_var.ROOT_DIR:
        return
    if not os.path.isdir(dst):
        os.makedirs(dst)
    all_files = os.listdir(src)
    for fname in all_files:
        fname_full = opj(src, fname)
        fname_dst = opj(dst, fname)
        if os.path.isdir(fname_full):
            backup_file(fname_full, fname_dst)
        elif fname.endswith('.py') or fname.endswith('.sh'):
            shutil.copy(fname_full, fname_dst)


def prepare_log_dir(log_name, log_dir=None):
    if len(log_name) == 0:
        log_name = datetime.now().strftime("%b%d_%H%M%S")
    if log_dir is None:
        log_dir = os.path.join(global_var.LOG_DIR, log_name)
    if not os.path.exists(log_dir):
        print('making %s' % log_dir)
        os.makedirs(log_dir)

    backup_dir = opj(log_dir, 'code')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    backup_file(global_var.ROOT_DIR, backup_dir)
    print("Backup code in {}".format(backup_dir))
    ckpt_dir = opj(log_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    return log_dir
