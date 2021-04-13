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


def _read_split_file(filepath):
    '''
    Read data split txt file provided for Robust Vision
    '''
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [x.strip() for x in trajs]
    return trajs


def fill_depth(depth):
    x, y = np.meshgrid(np.arange(depth.shape[1]).astype("float32"),
                       np.arange(depth.shape[0]).astype("float32"))
    xx = x[depth > 0]
    yy = y[depth > 0]
    zz = depth[depth > 0]

    grid = interpolate.griddata((xx, yy), zz.ravel(),
                                (x, y), method='nearest')
    return grid


def augument(images):
    # randomly shift gamma

    random_gamma = np.random.uniform(0.9, 1.1, size=1)
    images = 255.0 * ((images / 255.0) ** random_gamma)

    # randomly shift brightness
    random_brightness = np.random.uniform(0.8, 1.2, size=1)
    images *= random_brightness

    # randomly shift color
    random_colors = np.random.uniform(0.8, 1.2, size=[3])
    images *= np.reshape(random_colors, [1, 1, 1, 3])

    images = np.clip(images, 0.0, 255.0)

    return images


class ScannetDataset(data.Dataset):
    def __init__(self, dataset_path, split_txt=None, height=256, width=320, n_frames=5,
                 depth_min=0.1, depth_max=10., mode='train', reloadscan=False):
        super(ScannetDataset, self).__init__()
        self.dataset_path = dataset_path
        self.n_frames = n_frames
        self.height = height
        self.width = width
        self.depth_min = depth_min
        self.depth_max = depth_max

        self.reloadscan = reloadscan

        self.mode = mode  # train or test

        if os.path.exists(split_txt):
            self.scenes = _read_split_file(split_txt)
        else:
            self.scenes = sorted(os.listdir(self.dataset_path))

        self.build_dataset_index_train(r=self.n_frames)

        scale_w = self.width / 640.
        scale_h = self.height / 480.
        self.cam_intr = torch.tensor([[577.87 * scale_w, 0, 319.5 * scale_w],
                                      [0, 577.87 * scale_h, 239.5 * scale_h],
                                      [0, 0, 1]]).to(torch.float32)

        self.proc_totensor = m_preprocess.to_tensor()

    def __len__(self):
        return len(self.dataset_index)

    def shape(self):
        return [self.n_frames, self.height, self.width]

    def read_sample_train(self, index):
        data_blob = self.dataset_index[index]
        assert self.n_frames == data_blob['n_frames']

        images = []
        images_paths = []
        img_ids = []

        poses = []
        poses_paths = []
        pose_ids = []

        depths = []
        dmasks = []
        depths_paths = []
        depth_ids = []

        for i in range(self.n_frames):
            image = cv2.imread(data_blob['images'][i])

            img_id = re.findall(r'\d+', os.path.basename(data_blob['images'][i]))
            img_ids.append(img_id)

            images_paths.append(data_blob['images'][i])
            image = cv2.resize(image, (self.width, self.height))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

            # load pose
            pose = np.loadtxt(data_blob['poses'][i], delimiter=' ').astype(np.float32)

            pose_id = re.findall(r'\d+', os.path.basename(data_blob['poses'][i]))
            pose_ids.append(pose_id)

            poses.append(pose)
            poses_paths.append(data_blob['poses'][i])

            # load depth
            depth = cv2.imread(data_blob['depths'][i], cv2.IMREAD_ANYDEPTH)
            depth = cv2.resize(depth, (self.width, self.height))

            depth_id = re.findall(r'\d+', os.path.basename(data_blob['depths'][i]))
            depth_ids.append(depth_id)

            depth = (depth.astype(np.float32)) / 1000.0

            dmask = (depth >= self.depth_min) & (depth <= self.depth_max) & (np.isfinite(depth))
            depth[~dmask] = 0

            ratio = np.sum(np.float32(dmask)) / (self.width * self.height)

            assert ratio > 0.5

            depths.append(depth)
            dmasks.append(dmask)
            depths_paths.append(data_blob['depths'][i])

        images = np.stack(images, axis=0).astype(np.float32)
        poses = np.stack(poses, axis=0).astype(np.float32)

        assert np.all(np.isfinite(poses))
        assert (img_ids == pose_ids) & (img_ids == depth_ids)

        depths = np.stack(depths, axis=0).astype(np.float32)
        dmasks = np.stack(dmasks, axis=0)

        return images, poses, depths, dmasks, img_ids, images_paths

    def __getitem__(self, index):
        # images, poses, depths, dmasks, frameid = self.read_sample_train(index)

        flag = True
        while flag:
            try:
                images, poses, depths, dmasks, frameid, images_paths = self.read_sample_train(index)

                flag = False
            except:

                tmp = np.random.randint(0, self.__len__(), 1)[0]
                print("data load error!", index, "use:  ", tmp)
                index = tmp

        # it seems that augment will influence accuracy
        # do_augument = np.random.uniform(0, 1, size=1)
        # if do_augument < 0.5:
        #     images = augument(images)

        sample = {
            'imgs': torch.from_numpy(images).permute(0, 3, 1, 2).to(torch.float32),  # [N,3,H,W]
            'dmaps': torch.from_numpy(depths).unsqueeze(1).to(torch.float32),  # [N,1,H,W]
            'dmasks': torch.from_numpy(dmasks).unsqueeze(1),  # [N,1,H,W]
            'cam_poses': torch.from_numpy(poses).to(torch.float32),  # [N,4,4]
            'cam_intr': self.cam_intr.to(torch.float32),
            'img_path': images_paths
        }

        return sample

    def _load_scan(self, scan, interval, if_dump=True):
        """

        :param scan:
        :param interval: 2 if train mode; 10 if test mode
        :return:
        """
        scan_path = os.path.join(self.dataset_path, scan)

        datum_file = os.path.join(scan_path, 'scene.npy')

        # really need to sample scene more densely (skip every 2 frames not 4)

        if (not os.path.exists(datum_file)) or self.reloadscan:
            print("load ", datum_file, self.reloadscan, type(self.reloadscan))
            imfiles = glob.glob(os.path.join(scan_path, 'pose', '*.txt'))
            ixs = sorted([int(os.path.basename(x).split('.')[0]) for x in imfiles])

            poses = []
            for i in ixs[::interval]:
                posefile = os.path.join(scan_path, 'pose', '%d.txt' % i)
                pose = np.loadtxt(posefile, delimiter=' ').astype(np.float32)

                if ~np.all(np.isfinite(pose)):
                    break
                else:
                    poses.append(posefile)

            images = []
            for i in ixs[::interval]:
                imfile = os.path.join(scan_path, 'rgb', '%d.jpg' % i)
                images.append(imfile)

            depths = []
            for i in ixs[::interval]:
                depthfile = os.path.join(scan_path, 'depth', '%d.png' % i)
                depths.append(depthfile)

            valid_num = len(poses)

            scene_info = {
                "images": images[:valid_num],
                "depths": depths[:valid_num],
                "poses": poses
            }

            if if_dump:
                np.save(datum_file, scene_info)
            return scene_info

        else:
            return np.load(datum_file, allow_pickle=True).item()

    def build_dataset_index_train(self, r=4):
        self.dataset_index = []
        data_id = 0
        skip = r // 2

        for scan in self.scenes:
            scanid = int(re.findall(r'scene(.+?)_', scan)[0])

            scene_info = self._load_scan(scan, interval=10)
            images = scene_info["images"]
            depths = scene_info["depths"]
            poses = scene_info["poses"]

            for i in range(r, len(images) - r, skip):
                training_example = {}
                training_example['depths'] = depths[i - r:i + r + 1]
                training_example['images'] = images[i - r:i + r + 1]
                training_example['poses'] = poses[i - r:i + r + 1]
                training_example['n_frames'] = r
                training_example['id'] = data_id

                self.dataset_index.append(training_example)
                data_id += 1
