#!/usr/bin/env python
import os
import time
import copy
import random
import seaborn as sns
import numpy as np
import json
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim import lr_scheduler

import script
import argparse

# Argparser Arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/", help="True")
parser.add_argument('--save_dir', action='store', dest='save_dir', default='./', help='path of checkpoint')
parser.add_argument('--arch', action='store', dest='arch', default='vgg19', choices={"vgg19", "densenet161"}, help='network arch')
parser.add_argument('--learning_rate', action='store', nargs='?', default=0.001, type=float, dest='learning_rate', help='learning rate of the network type float')
parser.add_argument('--epochs', action='store', dest='epochs', default=10, type=int, help='number of epochs while training type int')
parser.add_argument('--hidden_units', action='store', nargs=1, default=4096, dest='hidden_units', type=int, help='Enter hidden units')
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='Set a switch to use GPU')
results = parser.parse_args()

data_dir = results.data_dir
save_dir = results.save_dir
arch = results.arch
hidden_units = results.hidden_units
epoch = results.epochs
lr = results.learning_rate
gpu = results.gpu
print_every = 30


if gpu==True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('GPU On')
else:
    print('CPU ON')
    device = 'cpu'
    
    
# Loading Dataset
data_transforms, directories, dataloaders, dataset_sizes, image_datasets = script.loading_data(data_dir)
class_to_idx = image_datasets['training_transforms'].class_to_idx
print("cudaorcpu_3")
for i in dataloaders:
    print("dataloaders ", dataloaders[i])

# Network Setup
model, input_size = script.make_model(arch, hidden_units)
criteria = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
epochs=epoch
model_ft = script.train_model(dataloaders,dataset_sizes, model, criteria, optimizer, sched, epochs, device)    

# Testing Model
script.check_accuracy_on_test(dataloaders, model, 'testing_transforms', True) 

# Saving Checkpoint
script.save_checkpoints(model, arch, lr, epochs, input_size, hidden_units, class_to_idx, save_dir)