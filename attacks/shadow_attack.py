# -*- coding: utf-8 -*-

# Implementation of the shadow attacks from the paper, 
# Shadows can be dangerous....

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
from pso import PSO
from shadow_utils import brightness
from shadow_utils import shadow_edge_blur
from shadow_utils import judge_mask_type
from shadow_utils import draw_shadow
from shadow_utils import load_mask
from shadow_utils import pre_process_image
from collections import Counter
from torchvision import transforms
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..')) 
# Directory indirection added from evaluate.py by me ^^
from models.resnet import ResNet
from common import utils


with open('params.json', 'rb') as f:
    params = json.load(f)
    class_n = params['GTSRB']['class_n']
    # class_n_lisa = params['LISA']['class_n']
    device = params['device']
    position_list, mask_list = load_mask()

parser = argparse.ArgumentParser(description="Adversarial attack by shadow")
parser.add_argument("--shadow_level", type=float, default=0.43,
                    help="shadow coefficient k")
parser.add_argument("--attack_db", type=str, default="GTSRB",
                    help="the target dataset should be specified for a digital attack")
parser.add_argument("--attack_type", type=str, default="digital",
                    help="digital attack or physical attack")
parser.add_argument("--image_path", type=str, default="./xxx",
                    help="a file path to the target image should be specified for a physical attack")
parser.add_argument("--mask_path", type=str, default="./xxx",
                    help="a file path to the mask should be specified for a physical attack")
parser.add_argument("--image_label", type=int, default=0,
                    help="a ground truth should be specified for a physical attack")
parser.add_argument("--polygon", type=int, default=3,
                    help="shadow shape: n-sided polygon")
parser.add_argument("--n_try", type=int, default=5,
                    help="n-random-start strategy: retry n times")
parser.add_argument("--target_model", type=str, default="normal",
                    help="attack normal model or robust model")
parser.add_argument('--n_classes', type=int, default=43,    # No. of classes added by me
                        help='Number of classes in data')                     
parser.add_argument('--model_type', type=str, default = 'normal', # required=True,   # Type of attack param added by me
                        help='Specify the type of model "normal" or "adversarial"')                     


args = parser.parse_args()
shadow_level = args.shadow_level
target_model = args.target_model
attack_db = args.attack_db
attack_type = args.attack_type
image_path = args.image_path
mask_path = args.mask_path
image_label = args.image_label
polygon = args.polygon
n_try = args.n_try
n_classes = args.n_classes
model_type = args.model_type


model = ResNet(n_classes, [3, 32, 32], channels=12, blocks=[
                              3, 3, 3], clamp=True).to(device)

# assert attack_db in ['GTSRB', 'LISA'] Using default in every case
if model_type == "normal":   
    model.load_state_dict(
        torch.load('./experiments/model_complete_120.pt',
                   map_location=torch.device(device)))
    pre_process = transforms.Compose([transforms.ToTensor()])
else:
    model.load_state_dict(
        torch.load('./experiments/adversarial_model_complete_140.pt',
                   map_location=torch.device(device)))
    pre_process = transforms.Compose([
        pre_process_image, transforms.ToTensor()])
model.eval()


assert attack_type in ['digital', 'physical']
if attack_type == 'digital':
    particle_size = 10
    iter_num = 100
    x_min, x_max = -16, 48
    max_speed = 1.5
else:
    particle_size = 10
    iter_num = 200
    x_min, x_max = -112, 336
    max_speed = 10.
    n_try = 1


def attack(attack_image, label, coords, targeted_attack=False, physical_attack=False, **parameters):
    r"""
    Physical-world adversarial attack by shadow.
    Args:
        attack_image: The image to be attacked.
        label: The ground-truth label of attack_image.
        coords: The coordinates of the points where mask == 1.
        targeted_attack: Targeted / Non-targeted attack.
        physical_attack: Physical / digital attack.
    Returns:
        adv_img: The generated adversarial image.
        succeed: Whether the attack is successful.
        num_query: Number of queries.
    """
    num_query = 0
    succeed = False
    global_best_solution = float('inf')
    global_best_position = None

    for attempt in range(n_try):

        if succeed:
            break

        print(f"try {attempt + 1}:", end=" ")

        pso = PSO(polygon*2, particle_size, iter_num, x_min, x_max, max_speed,
                  shadow_level, attack_image, coords, model, targeted_attack,
                  physical_attack, label, pre_process, **parameters)
        best_solution, best_pos, succeed, query = pso.update_digital() \
            if not physical_attack else pso.update_physical()

        if targeted_attack:
            best_solution = 1 - best_solution
        print(f"Best solution: {best_solution} {'succeed' if succeed else 'failed'}")
        if best_solution < global_best_solution:
            global_best_solution = best_solution
            global_best_position = best_pos
        num_query += query

    adv_image, shadow_area = draw_shadow(
        global_best_position, attack_image, coords, shadow_level)
    adv_image = shadow_edge_blur(adv_image, shadow_area, 3)

    return adv_image, succeed, num_query


def attack_digital():

    save_dir = f'./images/adv_img/{attack_db}/{int(shadow_level*100)}'
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        for name in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, name))

    with open(f'./dataset/{attack_db}/test.pkl', 'rb') as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data['data'], test_data['labels']

    # images = utils.read_hdf5('./kaggle_gtsrb/kaggle_gtsrb/test_images.h5')
    # labels = utils.read_hdf5('./kaggle_gtsrb/kaggle_gtsrb/test_labels.h5')

    # arrayBrightness = -1 * np.ones(shape=(images.shape[0], 1))

    for index in range(images.shape[0]):
        mask_type = judge_mask_type(attack_db, labels[index])

        # arrayBrightness[index] = brightness(images[index], mask_list[mask_type])
        
        if brightness(images[index], mask_list[mask_type]) >= 120:  
            adv_img, success, num_query = attack(
                images[index], labels[index], position_list[mask_type])
            cv2.imwrite(f"{save_dir}/{index}_{labels[index]}_{num_query}_{success}.bmp", adv_img)

    # Code below used to calculate and visualize the mean brightness of train and test data

    # plt.hist(arrayBrightness, bins=64)
    # plt.savefig('BrightnessHistogram')    
    # flattened = arrayBrightness.flatten()
    # print('The mean brightness of the data is:', statistics.mean(flattened))   
    # The mean brightness of the kaggle training data is: 33.70918578109069
    # The mean brightness of the kaggle testing data is: 33.60315009477303

    print("Attack finished! Success rate: ", end='')
    print(Counter(map(lambda x: x[:-4].split('_')[-1],
                      os.listdir(save_dir)))['True'] / len(os.listdir(save_dir)))


# def attack_physical():

#     global position_list

#     mask_image = cv2.resize(
#         cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (224, 224))
#     target_image = cv2.resize(
#         cv2.imread(image_path), (224, 224))
#     pos_list = np.where(mask_image.sum(axis=2) > 0)

#     # EOT is included in the first stage
#     adv_img, _, _ = attack(target_image, image_label, pos_list,
#                            physical_attack=True, transform_num=10)
    
#     cv2.imwrite('./tmp/temp.bmp', adv_img)
#     if attack_db == 'LISA':
#         predict, failed = lisa.test_single_image(
#             './tmp/temp.bmp', image_label, target_model == "robust")
#     else:
#         predict, failed = gtsrb.test_single_image(
#             './tmp/temp.bmp', image_label, target_model == "robust")
#     if failed:
#         print('Attack failed! Try to run again.')

#     # Predict stabilization
#     adv_img, _, _ = attack(target_image, image_label, pos_list, targeted_attack=True,
#                            physical_attack=True, target=predict, transform_num=10)

#     cv2.imwrite('./tmp/adv_img.png', adv_img)
#     if attack_db == 'LISA':
#         predict, failed = lisa.test_single_image(
#             './tmp/adv_img.png', image_label, target_model == "robust")
#     else:
#         predict, failed = gtsrb.test_single_image(
#             './tmp/adv_img.png', image_label, target_model == "robust")
#     if failed:
#         print('Attack failed! Try to run again.')
#     else:
#         print('Attack succeed! Try to implement it in the real world.')

#     cv2.imshow("Adversarial image", adv_img)
#     cv2.waitKey(0)


if __name__ == '__main__':

    attack_digital() # if attack_type == 'digital' else attack_physical()