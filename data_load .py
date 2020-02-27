import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

    

    
# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy=  image_copy / 255.0
            
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
#         key_pts_copy = (key_pts_copy - 100)/50.0
        key_pts_copy = (key_pts_copy - 48) / 48.0 # NaimishNet paper p3 Data Pre-processing

        return {'image': image_copy, 'keypoints': key_pts_copy}



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_pickle_file,label_pickle_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images= pickle.load(open(image_pickle_file,"rb"))
        self.labels= pickle.load(open(label_pickle_file,"rb"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if True:
            image=self.images[idx]
            keypoints=self.labels[idx]
            sample = {'image': image, 'keypoints': keypoints}

            if self.transform:
                sample = self.transform(sample)

            return sample
        
            
            #print(e)
 



# Perform data augmentation on train dataset by first scaling and then random cropping

