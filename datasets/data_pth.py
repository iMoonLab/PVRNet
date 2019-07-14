import os
import torch
import models
import numpy as np
import os.path as osp
from PIL import Image
from itertools import groupby
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import normal_pc, STATUS_TEST, STATUS_TRAIN, pc_aug_funs
import pdb

name_list = ['night_stand', 'range_hood', 'plant', 'chair', 'tent', 'curtain', 'piano', 'dresser', 'desk', 'bed',
             'sink', 'laptop', 'flower_pot', 'car', 'stool', 'vase', 'monitor', 'airplane', 'stairs', 'glass_box',
             'bottle', 'guitar', 'cone', 'toilet', 'bathtub', 'wardrobe', 'radio', 'person', 'xbox', 'bowl', 'cup',
             'door', 'tv_stand', 'mantel', 'sofa', 'keyboard', 'bookshelf', 'bench', 'table', 'lamp']

def get_info(shapes_dir, isView=False):
    names_dict = {}
    if isView:
        for shape_dir in shapes_dir:
            name = '_'.join(osp.split(shape_dir)[1].split('.')[0].split('_')[:-1])
            if name in names_dict:
                names_dict[name].append(shape_dir)
            else:
                names_dict[name] = [shape_dir]
    else:
        for shape_dir in shapes_dir:
            name = osp.split(shape_dir)[1].split('.')[0]
            names_dict[name] = shape_dir

    return names_dict


class pc_data(Dataset):
    def __init__(self, pc_root, status='train', pc_input_num=1024):
        super(pc_data, self).__init__()

        self.status = status
        self.pc_list = []
        self.lbl_list = []
        self.pc_input_num = pc_input_num

        if status == STATUS_TRAIN:
            npy_list = glob(osp.join(pc_root, '*', 'train', '*.npy'))
        else:
            npy_list = glob(osp.join(pc_root, '*', 'test', '*.npy'))
        names_dict = get_info(npy_list, isView=False)

        for name, _dir in names_dict.items():
            self.pc_list.append(_dir)
            self.lbl_list.append(name_list.index('_'.join(name.split('_')[:-1])))

        print(f'{status} data num: {len(self.pc_list)}')

    def __getitem__(self, idx):
        lbl = self.lbl_list[idx]
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc = normal_pc(pc)
        if self.status == STATUS_TRAIN:
            pc = pc_aug_funs(pc)
        pc = np.expand_dims(pc.transpose(), axis=2)
        return torch.from_numpy(pc).float(), lbl

    def __len__(self):
        return len(self.pc_list)


class view_data(Dataset):
    def __init__(self, view_root, base_model_name=models.ALEXNET, status=STATUS_TRAIN):
        super(view_data, self).__init__()

        self.status = status
        self.view_list = []
        self.lbl_list = []

        if base_model_name in (models.ALEXNET, models.VGG13, models.VGG13BN, models.VGG11BN, models.RESNET50):
            self.img_sz = 224
        elif base_model_name in (models.RESNET101):
            self.img_sz = 227
        elif base_model_name in models.INCEPTION_V3:
            self.img_sz = 299
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize(self.img_sz),
            transforms.ToTensor()
        ])

        if status==STATUS_TRAIN:
            jpg_list = glob(osp.join(view_root, '*', 'train', '*.jpg'))
        else:
            jpg_list = glob(osp.join(view_root, '*', 'test', '*.jpg'))
        names_dict = get_info(jpg_list, isView=True)

        for name, _dirs in names_dict.items():
            self.view_list.append(_dirs)
            self.lbl_list.append(name_list.index('_'.join(name.split('_')[:-1])))

        self.view_num = len(self.view_list[0])

        print(f'{status} data num: {len(self.view_list)}')

    def __getitem__(self, idx):
        views = [self.transform(Image.open(v)) for v in self.view_list[idx]]
        return torch.stack(views).float(), self.lbl_list[idx]

    def __len__(self):
        return len(self.view_list)


class pc_view_data(Dataset):
    def __init__(self, pc_root, view_root, base_model_name=models.ALEXNET, status='train', pc_input_num=1024):
        super(pc_view_data, self).__init__()

        self.status = status
        self.view_list = []
        self.pc_list = []
        self.lbl_list = []
        self.pc_input_num = pc_input_num

        if base_model_name in (models.ALEXNET, models.VGG13, models.VGG13BN, models.VGG11BN):
            self.img_sz = 224
        elif base_model_name in (models.RESNET50, models.RESNET101):
            self.img_sz = 224
        elif base_model_name in models.INCEPTION_V3:
            self.img_sz = 299
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
            transforms.Resize(self.img_sz),
            transforms.ToTensor()
        ])

        if status == STATUS_TRAIN:
            jpg_list = glob(osp.join(view_root, '*', 'train', '*.jpg'))
            npy_list = glob(osp.join(pc_root, '*', 'train', '*.npy'))
        else:
            jpg_list = glob(osp.join(view_root, '*', 'test', '*.jpg'))
            npy_list = glob(osp.join(pc_root, '*', 'test', '*.npy'))
        pc_dict = get_info(npy_list, isView=False)
        view_dict = get_info(jpg_list, isView=True)

        for name in pc_dict.keys():
            self.view_list.append(view_dict[name])
            self.pc_list.append(pc_dict[name])
            self.lbl_list.append(name_list.index('_'.join(name.split('_')[:-1])))

        self.view_num = len(self.view_list[0])

        print(f'{status} data num: {len(self.view_list)}')

    def __getitem__(self, idx):
        names = osp.split(self.pc_list[idx])[1].split('.')[0]
        views = [self.transform(Image.open(v)) for v in self.view_list[idx]]
        lbl = self.lbl_list[idx]
        pc = np.load(self.pc_list[idx])[:self.pc_input_num].astype(np.float32)
        pc = normal_pc(pc)
        if self.status == STATUS_TRAIN:
            pc = pc_aug_funs(pc)
        pc = np.expand_dims(pc.transpose(), axis=2)
        # return torch.stack(views).float(), torch.from_numpy(pc).float(), lbl, names
        return torch.stack(views).float(), torch.from_numpy(pc).float(), lbl

    def __len__(self):
        return len(self.pc_list)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

class ModelNet(Dataset):
    def __init__(self, data_root, status=STATUS_TRAIN, img_size=224):
        super(ModelNet, self).__init__()
        self.data_root = data_root
        self.status = status
        self.img_size = img_size
        self.views_list = []
        self.label_list = []
        for i, curr_category in enumerate(sorted(get_immediate_subdirectories(self.data_root))):
            if status == STATUS_TEST:
                working_dir = os.path.join(data_root, curr_category, 'test')
            elif status == STATUS_TRAIN:
                working_dir = os.path.join(data_root, curr_category, 'train')
            else:
                raise NotImplementedError
            all_img_list = glob(working_dir + "/*.jpg")
            append_views_list = [[v for v in g] for _, g in groupby(sorted(all_img_list), lambda x: x.split('_')[-2])]
            self.views_list += append_views_list
            self.label_list += [i] * len(append_views_list)
        assert len(self.views_list) == len(self.label_list)
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        views = [self.transform(Image.open(v)) for v in self.views_list[index]]
        return torch.stack(views), self.label_list[index]

    def __len__(self):
        return len(self.views_list)

if __name__ == '__main__':
    # vd = view_data(cfg, state='test', batch_size=8, shuffle=True)
    # batch_len = len(vd)
    # imgs, lbls = vd.get_batch(307)
    # print(batch_len)
    # print(imgs.shape)
    # print(lbls)
    # Image.fromarray((imgs[0][0]*255).astype(np.uint8)).show()


    pvd = pc_view_data(status=STATUS_TEST)
    batch_len = len(pvd)
    # imgs, pcs, lbls = pvd.get_batch(307)
    # print(batch_len)
    # print(imgs.shape)
    # print(lbls)
    # Image.fromarray((imgs[0][0]*255).astype(np.uint8)).show()
    # utils.generate_pc.draw_pc(pcs[0])
