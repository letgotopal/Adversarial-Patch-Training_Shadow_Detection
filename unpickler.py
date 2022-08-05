import csv
import os
import statistics
import sys
import cv2
import json
from matplotlib import pyplot as plt
import torch
import pickle
import argparse
import numpy as np
from collections import Counter
from torchvision import transforms
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..')) 
# Directory indirection added from evaluate.py by me ^^
from models.resnet import ResNet
from common import utils

success = True

with open(f'./dataset/GTSRB/test.pkl', 'rb') as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data['data'], test_data['labels']

f = open('./dataset/GTSRB/Test.csv', 'w')
writer = csv.writer(f)

for index in range(images.shape[0]):
    cv2.imwrite(f"./dataset/GTSRB/Test/{index}_{labels[index]}.png", images[index])
    index_name = f'Test/{index}_{labels[index]}.png'
    row = labels[index], index_name
    writer.writerow(row)

f.close()

