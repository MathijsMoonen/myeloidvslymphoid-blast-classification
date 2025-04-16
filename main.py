# importing the libraries

from __future__ import print_function

import argparse

import os
import sys
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import cv2
import  glob
import time
import albumentations
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
### Internal Imports
from models.ResNext50 import Myresnext50
from train.train_classification import trainer_classification
from utils.utils import configure_optimizers
from Datasets.DataLoader import Img_DataLoader

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import glob

def main(args):

    img_dir = args.train_dir
    image_files = glob.glob(img_dir + '*/*.png')
    labels = [x.split('/')[-2] for x in image_files]

    # access all images
    X_train = glob.glob(os.path.join(args.train_dir,'*/*'))
    X_val = glob.glob(os.path.join(args.val_dir,'*/*'))

    labels = [x.split('/')[-2] for x in X_train]
    cell_types = set(labels)

    cell_types = list(cell_types)
    cell_types.sort()

    cell_types_df = pd.DataFrame(cell_types, columns=['Cell_Types'])# converting type of columns to 'category'
    cell_types_df['Cell_Types'] = cell_types_df['Cell_Types'].astype('category')# Assigning numerical values and storing in another column
    cell_types_df['Cell_Types_Cat'] = cell_types_df['Cell_Types'].cat.codes



    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(cell_types_df[['Cell_Types_Cat']]).toarray())
    cell_types_df = cell_types_df.join(enc_df)

    # load model

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # Interesting! This worked for no reason haha
    if args.input_model == 'ResNeXt50':
        resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=args.pretrained)
        my_extended_model = Myresnext50(my_pretrained_model= resnext50_pretrained, num_classes = 23)

    ## Simple augumentation to improve the data generalibility

    transform_pipeline = albumentations.Compose(
        [
            albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        ]
    )

    trainer = trainer_classification(train_image_files=X_train, validation_image_files=X_val, model=my_extended_model,
                                     img_transform=transform_pipeline, init_lr=args.init_lr,
                                     lr_decay_every_x_epochs=args.lr_decay_every_x_epochs,

                                     weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs, gamma=args.gamma, df=cell_types_df,
                                     save_checkpoints_dir=args.save_checkpoints_dir)

    My_model = trainer.train(my_extended_model)


# Training settings
parser = argparse.ArgumentParser(description='Configurations for Model training')
parser.add_argument('--train_dir', type=str,
                    default='../../2022_05_18_cells_50_NORMAL/Cross_Validation/iteration_3/train/',
                    help='train data directory')

parser.add_argument('--val_dir', type=str,
                    default='../../2022_05_18_cells_50_NORMAL/Cross_Validation/iteration_3/val/',
                    help='train data directory')

parser.add_argument('--input_model', type=str,
                    default='ResNeXt50',
                    help='input model, the defulat is the pretrained model')

parser.add_argument('--pretrained', type=bool,
                    default=True,
                    help='the defulat is the pretrained model')

parser.add_argument('--init_lr', type=float,
                    default=0.001,
                    help='learning rate')

parser.add_argument('--weight_decay', type=float,
                    default=0.0005,
                    help='weight decay')

parser.add_argument('--gamma', type=float,
                    default=0.1,
                    help='gamma')

parser.add_argument('--epochs', type=float,
                    default=30,
                    help='epoch number')

parser.add_argument('--batch_size', type=int,
                    default=1024,
                    help='epoch number')

parser.add_argument('--lr_decay_every_x_epochs', type = int,
                    default=10,
                    help='learning rate decay per X step')

parser.add_argument('--save_checkpoints_dir', type = str,
                    default=None,
                    help='save dir')

args = parser.parse_args()

if __name__ == "__main__":
    main(args)
    print('Done')