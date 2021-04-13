import torch.utils.data as data
import numpy as np
from path import Path
import random
import os
import re
import torch
import cv2

import data.m_preprocess as m_preprocess

from natsort import natsorted
import re

import glob
import pdb

def check_pose(cam_pose):
    flag = np.all(np.isfinite(cam_pose))

    return flag


def prepare_seqs(scene_name, database, seq_length, interval, seq_inter, eval_all=False):
    """
    prepare the input images sequences
    :param scene_name:
    :param database:
    :param seq_length:
    :param interval: if interval=2, the real interval of sequence is 10
    :return:
    """
    img_fldr = os.path.join(database, scene_name)

    seqs = []

    img_names = natsorted(glob.glob(os.path.join(img_fldr, '*.color.*')))
    dmap_names = natsorted(glob.glob(os.path.join(img_fldr, '*.depth.*')))
    dmap_names = [x for x in dmap_names if not "colored" in x]

    _, img_ext = os.path.splitext(img_names[0])
    _, dmap_ext = os.path.splitext(dmap_names[0])

    num = len(img_names)

    if eval_all:
        start_indexs = interval
    else:
        start_indexs = 1

    for start_i in range(start_indexs):
        for i in range(start_i, num - seq_length * interval, seq_inter):
            flag = True
            samples = []
            for s_ in range(seq_length):
                s = s_ * interval
                img_name = img_names[i + s]
                index = int(re.findall(r'\d+', os.path.basename(img_name))[0])

                img_path = '%s/frame-%06d.color%s' % (img_fldr, index, img_ext)
                dmap_path = '%s/frame-%06d.depth%s' % (img_fldr, index, dmap_ext)
                pose_path = '%s/frame-%06d.pose.txt' % (img_fldr, index)

                flag = flag & check_pose(np.loadtxt(pose_path))

                sample = {'img_path': img_path,
                          'dmap_path': dmap_path,
                          'pose_path': pose_path}
                samples.append(sample)

            if flag:
                seqs.append(samples)

    return seqs


class SevenScenes(data.Dataset):
    def __init__(self, data_dir,
                 image_size=[320, 256],
                 depth_min=0.3,
                 depth_max=5.0,
                 ndepths=64,
                 seq_length=5,
                 frame_interval=5,
                 seq_inter=1,
                 eval_all=False,
                 reload=False):

        """
        Due to sequentially return samples from video sequences, batch_size should be fixed in advance
        :param training:
        :param data_dir:
        :param split_txt:
        :param image_size:
        :param depth_min:
        :param depth_max:
        """
        super(SevenScenes, self).__init__()

        self.data_dir = data_dir
        self.test_seqs_list = [('chess', 'seq-03'),
                               ('chess', 'seq-05'),
                               ('fire', 'seq-03'),
                               ('fire', 'seq-04'),
                               ('heads', 'seq-01'),
                               ('office', 'seq-02'),
                               ('office', 'seq-06'),
                               ('office', 'seq-07'),
                               ('office', 'seq-09'),
                               ('pumpkin', 'seq-01'),
                               ('pumpkin', 'seq-07'),
                               ('redkitchen', 'seq-03'),
                               ('redkitchen', 'seq-04'),
                               ('redkitchen', 'seq-06'),
                               ('redkitchen', 'seq-12'),
                               ('redkitchen', 'seq-14'),
                               ('stairs', 'seq-01'),
                               ('stairs', 'seq-04')]

        if reload:
            self.test_seqs_list = self.load_test_seqs_list()

        self.seq_length = seq_length

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_size = image_size

        self.frame_interval = frame_interval  # if 1, 5frames, if 2, 10 frames
        self.seq_inter = seq_inter * frame_interval
        self.eval_all = eval_all

        self.ndepths = ndepths

        self.get_cam_intrinsic()

        self.proc_normalize = m_preprocess.get_transform()
        self.proc_totensor = m_preprocess.to_tensor()

    def reset(self, scene, seq=None):
        if seq is not None:
            seqs = prepare_seqs(scene + "/" + seq, self.data_dir, seq_length=self.seq_length,
                                interval=self.frame_interval, seq_inter=self.seq_inter, eval_all=self.eval_all)
        else:
            seqs = prepare_seqs(scene, self.data_dir, seq_length=self.seq_length,
                                interval=self.frame_interval, seq_inter=self.seq_inter, eval_all=self.eval_all)

        self.seqs = seqs

    def load_test_seqs_list(self):
        test_seqs_list = []
        for dir in os.listdir(self.data_dir):
            if "consist" in dir:
                continue
            for subdir in os.listdir(os.path.join(self.data_dir, dir)):
                test_seqs_list.append((dir, subdir))
        return test_seqs_list
    def prepare_all_seqs(self):
        seqs = []
        for scene_name in self.self.test_seqs_list:
            seqs_ = prepare_seqs(scene_name[0] + "/" + scene_name[1], self.data_dir, seq_length=self.seq_length,
                                 interval=self.frame_interval)
            seqs += seqs_

        return seqs

    def get_cam_intrinsic(self):
        cam_intr = torch.tensor([[577.87, 0, 319.5],
                                 [0, 577.87, 239.5],
                                 [0, 0, 1]])

        scale_x = self.image_size[0] / 640.
        scale_y = self.image_size[1] / 480.

        cam_intr[0, :] *= scale_x
        cam_intr[1, :] *= scale_y

        self.cam_intr = cam_intr.to(torch.float32)

    def __getitem__(self, index):
        """
        :param index:
        :return:
            img, dmap, cam_pose
        """

        samples_path = self.seqs[index]
        img_raws = []
        imgs = []
        dmaps = []
        dmasks = []
        cam_poses = []
        img_paths = []

        for sample_path in samples_path:
            img_path = sample_path['img_path']
            img_paths.append(img_path)
            dmap_path = sample_path['dmap_path']
            pose_path = sample_path['pose_path']  # camera pose, camera_to_world

            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

                dmap = cv2.imread(dmap_path, -1) / 1000.
                # dmap = cv2.resize(dmap, (self.image_size[0], self.image_size[1]))

                cam_pose = np.loadtxt(pose_path)
                if not check_pose(cam_pose):
                    print("Nan in cam pose")
                    exit()
            except:
                print("data load error!!")
                print(img_path)
                print(dmap_path)

            dmask = (dmap >= self.depth_min) & (dmap <= self.depth_max) & (np.isfinite(dmap))
            dmap[~dmask] = 0

            # convert to tensor
            img_raws.append(torch.from_numpy(img).to(torch.float32))
            imgs.append(torch.from_numpy(img).to(torch.float32))
            dmaps.append(self.proc_totensor(dmap).to(torch.float32))
            dmasks.append(self.proc_totensor(dmask))
            cam_poses.append(self.proc_totensor(cam_pose).to(torch.float32))

        sample = {
            'imgs': torch.stack(imgs, dim=0).permute(0, 3, 1, 2).unsqueeze(0),  # [B, N,3,H,W]
            'imgs_raw': torch.stack(img_raws, dim=0).unsqueeze(0),  # [N,H,W,3]
            'dmaps': torch.stack(dmaps, dim=0).unsqueeze(0),  # [N,1,H,W]
            'dmasks': torch.stack(dmasks, dim=0).unsqueeze(0),  # [N,1,H,W]
            'cam_poses': torch.cat(cam_poses, dim=0).unsqueeze(0),  # [N,4,4]
            'cam_intr': self.cam_intr.unsqueeze(0),
            'img_path': img_paths
        }

        return sample

    def __len__(self):
        return len(self.seqs)
