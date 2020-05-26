from config import cfg
from datasets import TextDataset
from trainer import BertAttnGANTrainer as trainer

import os
import sys
import time
import datetime
import dateutil.tz
import numpy as np

import torch
import torchvision.transforms as transforms

if __name__ == "__main__":
    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** 2)
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = TextDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(dataloader)

    start_t = time.time()
    algo.train()
    end_t = time.time()
    print('Total time for training:', end_t - start_t)