import csv
import torchvision
import torch
import torch.nn as nn
import imp
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
import argparse
from PIL import Image
import os
from deepfool import deepfool

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/input_images')
parser.add_argument('--output_dir', default='/output_images')
args = parser.parse_args()

MainModel = imp.load_source('MainModel', "./models/tf_to_pytorch_resnet_v1_50.py")
pretrained_model = torch.load('./models/tf_to_pytorch_resnet_v1_50.pth')

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

#model_dimension = 224
#center_crop = 224
pre_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean,std = std)])

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)

transform_back = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, std)),
                        transforms.Normalize(mean=map(lambda x: -x, mean), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

pretrained_model = pretrained_model.cuda()
pretrained_model.eval()
pretrained_model.volatile = True

csvfile = open(os.path.join(args.input_dir, 'dev.csv'), 'r')
csvreader = csv.DictReader(csvfile)
for row in csvreader:
    filename = row['filename']
    trueLabel = row['trueLabel']
    print(filename)
    im_orig = Image.open(os.path.join(args.input_dir, filename))
    im = pre_transform(im_orig)
    #im = im.cuda()
    #out=pretrained_model(im[None,:,:,:])
    #print(out.shape)
    #print(im.shape)
    r, loop_i, label_orig, label_pert, pert_image = deepfool(im, pretrained_model)
    print("pert image shape:",pert_image.shape)
    img_attack = transform_back(pert_image.cpu()[0])
    img_attack.save(os.path.join(args.output_dir, filename))
