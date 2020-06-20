#!/usr/bin/env python
# Imports
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import script


parser = argparse.ArgumentParser()
parser.add_argument('--img_path', action='store', dest='img_path', help='where is your image?', required=True)
parser.add_argument('--ck_dir', action='store', dest='ck_path', help='checkpoint path', required=True)
parser.add_argument('--category_names', action="store", dest="category_names", default='cat_to_name.json')
parser.add_argument('--top_k', action="store", default=5, dest="top_k",  type=int)
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='Set a switch to use GPU')
results = parser.parse_args()

img_path = results.img_path
checkpoint_path = results.ck_path
category_names = results.category_names
top_k = results.top_k
gpu = results.gpu

if gpu==True:
    using_gpu = torch.cuda.is_available()
    device = 'gpu'
    print('GPU On');
else:
    print('GPU Off');
    device = 'cpu'

model = script.loading_checkpoint(checkpoint_path)
processed_image = script.process_image(img_path)
probs, classes = script.predict(processed_image, model, top_k, device)

# Label mapping
cat_to_name = script.labeling(category_names)
labels = []
for class_index in classes:
    labels.append(cat_to_name[str(class_index)])

# Converting from tensor to numpy-array
print('Name of class: ', labels[0])
print('Probability: ', probs)