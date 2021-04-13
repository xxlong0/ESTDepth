"""
ScanNet dataset return images indicated in txt file
"""
import numpy as np
import os
import cv2
import re
import csv
import glob
import random
import pickle

import torch.utils.data as data
import torch

import data.m_preprocess as m_preprocess

from scipy import interpolate

import re


def _read_testlist_file(filepath):
    '''
    Read test list file format: scenename, index
    '''
    f = open(filepath)
    testdata_list = []
    for line in f.readlines():
        try:
            items = line.strip().split()
            testdata_list.append((items[0], items[1]))
        except:
            print(items)

    return testdata_list


def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid


class ScannetTestDataset(data.Dataset):
    def __init__(self, dataset_path, test_listfile, height=256, width=320,
                 depth_min=0.1, depth_max=10.):
        super(ScannetTestDataset, self).__init__()
        self.dataset_path = dataset_path
        self.height = height
        self.width = width

        self.depth_min = depth_min
        self.depth_max = depth_max

        self.testdata_list = _read_testlist_file(test_listfile)

        self.cam_intr = torch.tensor([[577.87, 0, 319.5],
                                      [0, 577.87, 239.5],
                                      [0, 0, 1]]).to(torch.float32)

        self.proc_totensor = m_preprocess.to_tensor()

    def __len__(self):
        return len(self.dataset_index)

    def shape(self):
        return [self.n_frames, self.height, self.width]

    def read_sample_test(self, idx):
        scenename, index = self.testdata_list[idx]
        index = int(index)
        if index < 10:
            inds = [index + 10, index, index + 20, index + 30, index + 40]
        else:
            inds = [index - 10, index, index - 20, index - 30, index - 40]

        images = []
        images_paths = []
        for i in inds:
            image_path = os.path.join(self.dataset_path, scenename, 'rgb', str(i) + ".jpg")
            image = cv2.imread(image_path)
            images_paths.append(image_path)
            image = cv2.resize(image, (self.width, self.height))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        poses = []
        poses_paths = []
        for i in inds:
            pose_path = os.path.join(self.dataset_path, scenename, 'pose', str(i) + ".txt")
            pose = np.loadtxt(pose_path, delimiter=' ').astype(np.float32)
            poses.append(pose)
            poses_paths.append(pose_path)

        depths = []
        dmasks = []
        depths_paths = []
        for i in inds:
            depth_path = os.path.join(self.dataset_path, scenename, 'depth', str(i) + ".png")
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (self.width, self.height))

            depth = (depth.astype(np.float32)) / 1000.0

            dmask = (depth >= self.depth_min) & (depth <= self.depth_max) & (np.isfinite(depth))
            depth[~dmask] = 0

            depths.append(depth)
            dmasks.append(dmask)
            depths_paths.append(depth_path)

        images = np.stack(images, axis=0).astype(np.float32)
        poses = np.stack(poses, axis=0).astype(np.float32)

        assert np.all(np.isfinite(poses))

        depths = np.stack(depths, axis=0).astype(np.float32)
        dmasks = np.stack(dmasks, axis=0)

        return images, poses, depths, dmasks, scenename, index, images_paths

    def __getitem__(self, index):

        images, poses, depths, dmasks, scenename, index, images_paths = self.read_sample_test(index)

        sample = {
            'imgs': torch.from_numpy(images).permute(0, 3, 1, 2).to(torch.float32).unsqueeze(0),  # [N,3,H,W]
            'dmaps': torch.from_numpy(depths).unsqueeze(1).to(torch.float32).unsqueeze(0),  # [N,1,H,W]
            'dmasks': torch.from_numpy(dmasks).unsqueeze(1).unsqueeze(0),  # [N,1,H,W]
            'cam_poses': torch.from_numpy(poses).to(torch.float32).unsqueeze(0),  # [N,4,4]
            'cam_intr': self.cam_intr.to(torch.float32).unsqueeze(0),
            'img_path': images_paths,
            'scenename': scenename,
            'index': str(index)
        }

        return sample
