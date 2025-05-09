import albumentations
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils import data
import sys
import os

# Add parent directory to sys.path to import utils.py
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utils.utils import segment_cell


class Img_DataLoader(data.Dataset):
    def __init__(self, img_list='', labels=None, in_dim=3, split='train', transform=False, in_size=96, df=None, encoder=None,
                 if_external=False):
        super(Img_DataLoader, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.filelist = img_list
        self.in_size = in_size
        self.file_paths = img_list
        self.transform = transform
        self.df = df
        self.encoder = encoder
        self.if_external = if_external
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        sample = dict()
        img_path = self.file_paths[index]
        label = self.labels[index]
        
        # prepare image
        seg_img = cv2.imread(img_path)
        image = cv2.resize(seg_img, (256,256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.transform is not None:
            try:
                img = self.transform(image=image)["image"]
            except:
                assert 1 == 2, 'something wrong'
                

        # print(img.shape)
        # if self.if_external:
        #     img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        

        img = np.einsum('ijk->kij', img)

        #Extract the categorical value associated with the cell type (Lymphoid, Myeloid, Other) and respectively (0,  1, 2) 
        if self.split != "compute":  # Use compute if you only want the prediction results. if you do this, make sure you don't shuffle the data
            _label = self.df.loc[self.df['Cell_Types'] == label, ['Cell_Types_Cat']].to_numpy() # Extracts the correct categorical value
            sample["label"] = torch.from_numpy(_label).long()
            
        sample["image"] = torch.from_numpy(img).float()  # self.encoder(torch.from_numpy(img).float())
        sample["ID"] = img_path
        return sample