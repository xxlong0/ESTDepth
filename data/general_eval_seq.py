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


def check_pose(cam_pose):
    flag = np.all(np.isfinite(cam_pose))

    return flag


def prepare_seqs(scene_name, database, interval, start_i=0, Scannet_data=False):
    """
    prepare the input images sequences
    :param scene_name:
    :param database:
    :param interval: if interval=2, the real interval of sequence is 10
    :return:
    """
    img_fldr = os.path.join(database, scene_name)

    samples = []

    if not Scannet_data:
        img_names = natsorted(glob.glob(os.path.join(img_fldr, '*.color.*')))
        dmap_names = natsorted(glob.glob(os.path.join(img_fldr, '*.depth.*')))
        dmap_names = [x for x in dmap_names if not "colored" in x]
    else:
        img_names = natsorted(glob.glob(os.path.join(img_fldr, 'rgb/*')))
        dmap_names = natsorted(glob.glob(os.path.join(img_fldr, 'depth/*')))

    _, img_ext = os.path.splitext(img_names[0])
    _, dmap_ext = os.path.splitext(dmap_names[0])

    num = len(img_names)
    for i in range(start_i, num, interval):
        img_name = img_names[i]
        index = int(re.findall(r'\d+', os.path.basename(img_name))[0])

        if not Scannet_data:
            img_path = '%s/frame-%06d.color%s' % (img_fldr, index, img_ext)
            dmap_path = '%s/frame-%06d.depth%s' % (img_fldr, index, dmap_ext)
            pose_path = '%s/frame-%06d.pose.txt' % (img_fldr, index)
        else:
            img_path = '%s/rgb/%d%s' % (img_fldr, index, img_ext)
            dmap_path = '%s/depth/%d%s' % (img_fldr, index, dmap_ext)
            pose_path = '%s/pose/%d.txt' % (img_fldr, index)

        if check_pose(np.loadtxt(pose_path)):
            sample = {'img_path': img_path,
                      'dmap_path': dmap_path,
                      'pose_path': pose_path}
            samples.append(sample)

    return samples


class SevenScenesSeq(data.Dataset):
    def __init__(self, data_dir,
                 image_size=[320, 256],
                 depth_min=0.01,
                 depth_max=5.0,
                 ndepths=64,
                 seq_length=5,
                 frame_interval=5,
                 seq_inter=1,
                 start_i=0,
                 eval_dataset="scannet"):

        """
        Due to sequentially return samples from video sequences, batch_size should be fixed in advance
        :param training:
        :param data_dir:
        :param split_txt:
        :param image_size:
        :param depth_min:
        :param depth_max:
        """
        super(SevenScenesSeq, self).__init__()

        self.data_dir = data_dir

        if eval_dataset == "scannet":
            self.test_seqs_list = []
        elif eval_dataset == "sun3d":
            self.test_seqs_list = []
            self.load_test_seqs_list()
        elif eval_dataset == "7scenes":
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

        self.seq_length = seq_length

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_size = image_size

        self.start_i = start_i
        self.eval_dataset = eval_dataset

        self.frame_interval = frame_interval  # if 1, 5frames, if 2, 10 frames
        self.seq_inter = seq_inter * frame_interval

        self.ndepths = ndepths

        self.get_cam_intrinsic()

        self.proc_normalize = m_preprocess.get_transform()
        self.proc_totensor = m_preprocess.to_tensor()

    def reset(self, scene, seq=None):
        if seq is not None:
            seqs = prepare_seqs(scene + "/" + seq, self.data_dir, interval=self.frame_interval, start_i=self.start_i,
                                Scannet_data=self.eval_dataset == "scannet")
        else:
            seqs = prepare_seqs(scene, self.data_dir, interval=self.frame_interval, start_i=self.start_i,
                                Scannet_data=self.eval_dataset == "scannet")

        self.seqs = seqs

    def load_test_seqs_list(self):
        for dir in os.listdir(self.data_dir):
            if "consist" in dir:
                continue
            for subdir in os.listdir(os.path.join(self.data_dir, dir)):
                self.test_seqs_list.append((dir, subdir))

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

        sample_path = self.seqs[index]

        img_path = sample_path['img_path']

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
        img_raw = torch.from_numpy(img).to(torch.float32)
        img = self.proc_totensor(img).to(torch.float32)
        dmap = self.proc_totensor(dmap).to(torch.float32)
        dmask = self.proc_totensor(dmask)
        cam_pose = self.proc_totensor(cam_pose).to(torch.float32)

        sample = {
            'img': img_raw.permute(2, 0, 1).unsqueeze(0),
            'img_raw': img_raw.unsqueeze(0),
            'dmap': dmap.unsqueeze(0),
            'dmask': dmask.unsqueeze(0),
            'cam_pose': cam_pose.squeeze().unsqueeze(0),
            'cam_intr': self.cam_intr.unsqueeze(0),
            'img_path': img_path,
        }

        return sample

    def __len__(self):
        return len(self.seqs)
