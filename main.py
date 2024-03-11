import torch
from torch import nn
import os
from os import path
import torchvision
import torchvision.transforms as T
from typing import Sequence
from torchvision.transforms import functional as F
import numbers
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torchmetrics as TM
from dataclasses import dataclass
import dataclasses

# Convert a pytorch tensor into a PIL image
t2img = T.ToPILImage()
# Convert a PIL image into a pytorch tensor
img2t = T.ToTensor()

# Set the working (writable) directory.
working_dir = "/content/"

from utils.models import VisionTransformerForSegmentation
from utils.helpers import *
from utils.dataset import *
from utils.validation import IoULoss
from utils.test import *
from train_epoch import train_model

@dataclass
class VisionTransformerArgs:
    #Arguments to the VisionTransformerForSegmentation.
    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    out_channels: int = 2 # This is a binary segmentation problem. Only foreground and backgroud. So out_channels = 2
    embed_size: int = 768
    num_blocks: int = 12
    num_heads: int = 8
    dropout: float = 0.2

vit_args = dataclasses.asdict(VisionTransformerArgs())
vit = VisionTransformerForSegmentation(**vit_args)
m = vit
to_device(m)
my_dataset = SegmentationDataSet(image_lst, mask_lst)
train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(m.parameters(), lr=0.0004)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.8)

def train_loop(model, loader, epochs, optimizer, scheduler, save_path):
    epoch_i, epoch_j = epochs
    for i in range(epoch_i, epoch_j):
        epoch = i
        print(f"Epoch: {i:02d}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        train_model(model, loader, optimizer)
        if scheduler is not None:
            scheduler.step()
        print("")

if __name__ == '__main__':
  train_loop(m, train_loader, (1, 20), optimizer, scheduler, save_path=None)
