import random
import cv2
import SimpleITK as sitk
import numpy as np
from key_tics import KeyTics

class KZDataset():
    def __init__(self, path_0=None, path_1=None, path_m=None, num_frame=16, image_size=256, patch_size_tic=64, rawh=896, raww=704, typ='train', transform=None, rand=False):
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


        if typ == 'val':

            self.data_info = self.data_info_0 + self.data_info_1
        elif typ == 'train':

            self.data_info = self.data_info_0 + self.data_info_1

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
        '''SSIM based key tic selection'''
        tics = self.key_tic(ceus[::2,:,:,:], msk_pth)  # [wall,tumor,lung]
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