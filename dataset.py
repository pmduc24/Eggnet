from torch.utils.data import Dataset, DataLoader
import os 
import cv2
import torch
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
  
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode  # 'train' or 'valid'
        self.transform = transform
        
        # Set image and mask directories based on mode
        self.img_dir = os.path.join(root_dir, mode, 'images')
        self.mask_dir = os.path.join(root_dir, mode, 'masks')

        self.image_names = []
        self.mask_names = []

        # Collect image and mask filenames
        for n in os.listdir(self.img_dir):
            self.image_names.append(os.path.join(self.img_dir, n))
            self.mask_names.append(os.path.join(self.mask_dir, n.replace('.jpg', '.npy')))  

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

        mask_name = self.mask_names[idx]
        mask = np.load(mask_name)
        mask = torch.from_numpy(mask).unsqueeze(0).long()

        if self.transform:
            image = self.transform['Image'](image) 
            mask = self.transform['Mask'](mask)
       
        sample = {'image': image, 'mask': mask}
        return sample