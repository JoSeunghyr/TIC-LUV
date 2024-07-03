import random
import torch
import torch.nn as nn
import math
import cv2
import SimpleITK as sitk
import numpy as np
from transform import video_transforms, volume_transforms
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from key_tics import KeyTics

class KZDataset():
    def __init__(self, path_0=None, path_1=None, path_m=None, ki=0, K=5, num_frame=16, image_size=256, patch_size_tic=64, rawh=896, raww=704, typ='train', transform=None, rand=False):
        super().__init__()
        self.H = rawh   # for TICA patch size is 64
        self.W = raww  # for TICA patch size is 64
        self.image_size = image_size
        self.num_frame = num_frame
        self.data_info_0 = self.get_img_info(path_0)
        self.data_info_1 = self.get_img_info(path_1)
        if rand:
            random.seed(1)
            random.shuffle(self.data_info_0)
            random.shuffle(self.data_info_1)

        leng_0 = len(self.data_info_0)
        every_z_len_0 = leng_0 / K
        leng_1 = len(self.data_info_1)
        every_z_len_1 = leng_1 / K
        if typ == 'val':
            self.data_info_0 = self.data_info_0[math.ceil(every_z_len_0 * ki) : math.ceil(every_z_len_0 * (ki+1))]
            self.data_info_1 = self.data_info_1[math.ceil(every_z_len_1 * ki) : math.ceil(every_z_len_1 * (ki+1))]
            self.data_info = self.data_info_0 + self.data_info_1

            # self.data_samplr = video_transforms.Compose([UniformTemporalSubsample(int(self.num_frame/2))])  # (C, T, H, W)
            self.data_samplr = video_transforms.Compose([UniformTemporalSubsample(self.num_frame)])  # (C, T, H, W)
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize((2*self.image_size, self.image_size), interpolation='bilinear'),  # (T, H, W, C)
                #video_transforms.CenterCrop(size=(self.H, self.W)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif typ == 'train':
            self.data_info_0 = self.data_info_0[: math.ceil(every_z_len_0 * ki)] + self.data_info_0[math.ceil(every_z_len_0 * (ki+1)):]
            self.data_info_1 = self.data_info_1[: math.ceil(every_z_len_1 * ki)] + self.data_info_1[math.ceil(every_z_len_1 * (ki+1)):]
            self.data_info = self.data_info_0 + self.data_info_1

            # self.data_samplr = video_transforms.Compose([UniformTemporalSubsample(int(self.num_frame / 2))])  # (C, T, H, W)
            self.data_samplr = video_transforms.Compose([UniformTemporalSubsample(self.num_frame)])  # (C, T, H, W)

            self.data_transform = video_transforms.Compose([
                video_transforms.Resize((2*self.image_size, self.image_size), interpolation='bilinear'),  # (T, H, W, C)
                #video_transforms.CenterCrop(size=(self.H, self.W)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        print(len(self.data_info))
        self.is_transf = transform
        self.key_tic = KeyTics(patch_size=patch_size_tic, rawh=rawh, raww=raww)
        self.patch_size_tic = patch_size_tic
        self.path_m = path_m

    def __getitem__(self, index):
        img_pth, label, cp = self.data_info[index]
        cp = [int(i) for i in cp]
        cp = np.array(cp)
        img = sitk.ReadImage(img_pth)
        img = sitk.GetArrayFromImage(img)  # T,H,W,C
        ceus = np.zeros((img.shape[0], self.H, self.W, img.shape[3]), np.uint8)
        for ii in range(img.shape[0]):
            img_resize = cv2.resize(img[ii, :, int(img.shape[2]/2):, :], (self.W, self.H))
            ceus[ii, :, :, :] = img_resize
        patient = img_pth.split('\\')[-1][:-4].split('_')[0]
        msk_pth = self.path_m +'\\' + patient + '.nii.gz'
        if self.is_transf:
            img = torch.tensor(img).permute(3, 0, 1, 2)
            'frame_DMUV = frame_TICA/2'
            # img = self.data_samplr(img)
            # img = np.array(img.permute(1, 2, 3, 0))  # T,H,W,C
            # img = self.data_transform(img)  # C,T,H,W
            'frame_DMUV = frame_TICA'
            img = self.data_samplr(img)
            img = np.array(img.permute(1, 2, 3, 0))  # T,H,W,C
            img = self.data_transform(img)  # C,T,H,W
        img = img.transpose(0, 1)  # T,C,H,W
        '''SSIM based key tic selection'''
        tics = self.key_tic(ceus, msk_pth)  # [wall,tumor,lung]
        tics = np.array(tics)
        return img, tics, label, patient, cp

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(csv_path):
        data_info = []
        data = open(csv_path, 'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.replace(",", " ").replace("\n", "")
            data_line = data_line.split()
            img_pth = data_line[0]
            label = int(data_line[1])
            cp = data_line[2:6]  # clinical information
            data_info.append((img_pth, label, cp))
        return data_info