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

using_gpu = torch.cuda.is_available()
dt={}

def loading_data(data_dir):
    print(type(data_dir))
    for i in data_dir:
        print(i)
    train_dir = data_dir[0] + '/train'
    valid_dir = data_dir[0] + '/valid'
    test_dir = data_dir[0] + '/test'


    data_transforms = {
    'training_transforms': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'validation_transforms': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    'testing_transforms': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
    }

    directories = {'training_transforms': train_dir, 
        'validation_transforms': valid_dir, 
        'testing_transforms' : test_dir}


    image_datasets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x]) for x in        ['training_transforms', 'validation_transforms', 'testing_transforms']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['training_transforms', 'validation_transforms', 'testing_transforms']}
    dataset_sizes = {x: len(image_datasets[x])for x in ['training_transforms', 'validation_transforms', 'testing_transforms']}
    class_names = image_datasets['training_transforms'].classes
    dt={x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['training_transforms', 'validation_transforms', 'testing_transforms']}
    
    print("cudaorcpu_2")
    for i in dt:
        print("dataloaders ", dt[i])
    return data_transforms, directories, dataloaders, dataset_sizes, image_datasets

def make_model(arch, hidden_units):
    if arch=="vgg19":
        model = models.vgg19(pretrained=True)
        input_size = 25088
    else:
        model = models.densenet161(pretrained=True)
        input_size = 2208
    output_size = 102
    for param in model.parameters():
        param.requires_grad = False
    hidden_layer =hidden_units
    classifier =nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_layer)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_layer, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    cudaorcpu= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("cudaorcpu" , cudaorcpu)
    model.classifier = classifier
    return model, input_size


def train_model(dt,dz, model, criteria, optimizer, scheduler,num_epochs=25, device='cuda'):
    since = time.time()
    print("cudaorcpu_")
    for i in dt:
        print("dataloaders ", dt[i])
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    if device and torch.cuda.is_available():
        print('GPU')
        model.cuda()
    elif device and torch.cuda.is_available() == False:
        print('no gpu was found..')
    else:
        print('CPU')  
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Looping for every epoch 
        for phase in ['training_transforms', 'validation_transforms']:
            if phase == 'training_transforms':
                scheduler.step()
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0
       
            # for everydata we go through              
            for inputs, labels in dt[phase]:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()

                # tracking forward
                with torch.set_grad_enabled(phase == 'training_transforms'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criteria(outputs, labels)

                    # tracking backward
                    if phase == 'training_transforms':
                        loss.backward()
                        optimizer.step()

                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dz[phase]
            epoch_accuracy = running_corrects.double() / dz[phase]

            print('{} Loss:{:.3f} Accuracy:{:.3f}'.format(
                phase, epoch_loss, epoch_accuracy))

            
            if phase == 'validation_transforms' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    
    print('Total training time is {:.1f}m {:.1f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best value Accuracy: {:3f}'.format(best_accuracy))

    # load weights
    model.load_state_dict(best_model_wts)
    return model    
    
def check_accuracy_on_test(dt, model, data, cuda=False):
    model.eval()
    model.to(device='cuda') 
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in (dt[data]):
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Network accuracy on the test images: %d %%' % (100 * correct / total))
    
def save_checkpoints(model,arch, lr, epochs, input_size, hidden_units, class_to_idx, checkpoint_path):
    model.class_to_idx = class_to_idx
    model.cpu()
    
    state = {'structure': arch,
            'hidden_layer': hidden_units,
            'learning_rate': lr,
            'input_size': input_size,
            'state_dict': model.state_dict(),
            'epochs': epochs,
            'class_to_idx': model.class_to_idx}
    torch.save(state, checkpoint_path + 'classifier.pth') 
    print("checkpoint file ", checkpoint_path)   
 


def loading_checkpoint(path):
    
    # Loading the parameters
    state = torch.load(path)
    lr = state['learning_rate']
    input_size = state['input_size']
    structure = state['structure']
    hidden_units = state['hidden_layer']
    epochs = state['epochs']
    
    # Building the model from checkpoints
    model,_ = make_model(structure, hidden_units)
    model.class_to_idx = state['class_to_idx']
    model.load_state_dict(state['state_dict'])
    return model


# Inference for classification
def process_image(image):
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img

# Labeling
def labeling(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

# Class Prediction
def predict(processed_image, model, topk, device):    
    # Implement the code to predict the class from an image file
    model.eval()
    model.cpu()
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    
    with torch.no_grad():
        output = model.forward(processed_image)
        probs, classes = torch.topk(input=output, k=topk)
        top_prob = probs.exp()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in classes.cpu().numpy()[0]]
        
    return top_prob, top_classes