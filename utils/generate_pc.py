import os
import bisect
import config
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from math import isnan
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D


def des(a, b):
    return np.linalg.norm(a - b)


def get_info(shape_dir):
    splits = shape_dir.split('/')
    class_name = splits[-3]
    set_name = splits[-2]
    file_name = splits[-1].split('.')[0]
    return class_name, set_name, file_name


def random_point_triangle(a, b, c):
    r1 = np.random.random()
    r2 = np.random.random()
    p = np.sqrt(r1) * (r2 * c + b * (1-r2)) + a * (1-np.sqrt(r1))
    return p


def triangle_area(p1, p2, p3):
    a = des(p1, p2)
    b = des(p1, p3)
    c = des(p2, p3)
    p = (a+b+c)/2.0
    area = np.sqrt(p*(p-a)*(p-b)*(p-c))
    if isnan(area):
        # print('find nan')
        area = 1e-6
    return area


def uniform_sampling(points, faces, n_samples):
    sampled_points = []
    total_area = 0
    cum_sum = []
    for _idx, face in enumerate(faces):
        total_area += triangle_area(points[face[0]], points[face[1]], points[face[2]])
        if isnan(total_area):
            print('find nan')
        cum_sum.append(total_area)

    for _idx in range(n_samples):
        tmp = np.random.random()*total_area
        face_idx = bisect.bisect_left(cum_sum, tmp)
        pc = random_point_triangle(points[faces[face_idx][0]],
                                   points[faces[face_idx][1]],
                                   points[faces[face_idx][2]])
        sampled_points.append(pc)
    return np.array(sampled_points)


def resize_pc(pc, L):
    """
    normalize point cloud in range L
    :param pc: type list
    :param L:
    :return: type list
    """
    pc_L_max = np.sqrt(np.sum(pc ** 2, 1)).max()
    return pc/pc_L_max*L


def normal_pc(pc):
    """
    normalize point cloud in range L
    :param pc: type list
    :return: type list
    """
    pc_mean = pc.mean(axis=0)
    pc = pc - pc_mean
    pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
    pc = pc/pc_L_max
    return pc


def get_pc(shape, point_each):
    points = []
    faces = []
    with open(shape, 'r') as f:
        line = f.readline().strip()
        if line == 'OFF':
            num_verts, num_faces, num_edge = f.readline().split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)
        else:
            num_verts, num_faces, num_edge = line[3:].split()
            num_verts = int(num_verts)
            num_faces = int(num_faces)

        for idx in range(num_verts):
            line = f.readline()
            point = [float(v) for v in line.split()]
            points.append(point)

        for idx in range(num_faces):
            line = f.readline()
            face = [int(t_f) for t_f in line.split()]
            faces.append(face[1:])

    points = np.array(points)
    pc = resize_pc(points, 10)
    pc = uniform_sampling(pc, faces, point_each)

    pc = normal_pc(pc)

    return pc


def generate(raw_off_root, vis_pc=False, num_pc_each=2018):
    shape_all = glob(osp.join(raw_off_root, '*', '*', '*.off'))
    shuffle(shape_all)
    cnt = 0
    for shape in tqdm(shape_all):
        class_name, set_name, file_name = get_info(shape)
        new_folder = osp.join(config.pc_net.data_root, class_name, set_name)
        new_dir = osp.join(new_folder, file_name)
        if osp.exists(new_dir+'.npy'):
            if vis_pc and not osp.exists(new_dir+'.jpg'):
                pc = np.load(new_dir+'.npy')
                draw_pc(pc, show=False, save_dir=new_dir+'.jpg')
        else:
            pc = get_pc(shape, num_pc_each)
            if not osp.exists(new_folder):
                os.makedirs(new_folder)
            np.save(new_dir+'.npy', pc)
            if vis_pc:
                if cnt%10==0:
                    draw_pc(pc, show=False, save_dir=new_dir+'.jpg')
                cnt += 1


def draw_pc(pc, show=True, save_dir=None):
    ax = plt.figure().add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='.')
    ax.grid(False)
    # ax.axis('off')
    if show:
        plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)


if __name__ == '__main__':
    generate('/repository/Modelnet40')
    # file_name = '/home/fyf/data/pc_ModelNet40/airplane/train/airplane_0165.npy'
    # pc = np.load(file_name)
    # draw_pc(pc)


