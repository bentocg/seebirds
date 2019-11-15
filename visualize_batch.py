import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms

from utils.dataloaders import ImageFolderTrainDet
from utils.dataloaders.transforms_det import ShapeTransform


def imshow(inp, title=None, rgb=True):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(title, dpi=500)
    if not rgb:
        plt.imshow(inp, cmap='gray')
    else:
        plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# create dataloader
training_set = 'training_set_vanilla'
data_dir = './training_sets/{}'.format(training_set)
arch_input_size = 512

data_transforms = {
    'training': {'shape_transform': ShapeTransform(arch_input_size, train=True),
                 'int_transform': transforms.Compose([
                     transforms.ColorJitter(brightness=np.random.choice([0, 1]) * 0.05,
                                            contrast=np.random.choice([0, 1]) * 0.05),
                     transforms.ToTensor(),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])},
    'validation': {'shape_transform': ShapeTransform(arch_input_size, train=False),
                   'int_transform': transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])},
}

image_datasets = {x: ImageFolderTrainDet(root=os.path.join(data_dir, x),
                                         shape_transform=data_transforms[x]['shape_transform'],
                                         int_transform=data_transforms[x]['int_transform'],
                                         training_set=training_set)
                  for x in ['training', 'validation']}






# Get a batch of training data
dataloaders = {"training": torch.utils.data.DataLoader(image_datasets["training"],
                                                       batch_size=4,
                                                       num_workers=1,
                                                       shuffle=True),
               "validation": torch.utils.data.DataLoader(image_datasets["validation"],
                                                         batch_size=4,
                                                         num_workers=1,
                                                         shuffle=True)}

# inputs contains 4 images because batch_size=4 for the dataloaders
inputs, counts, locations = next(iter(dataloaders['validation']))

# Make a grid from batch
out_img = torchvision.utils.make_grid(inputs)
out_location = torchvision.utils.make_grid(locations)

imshow(out_img, title='input')
imshow(out_location, title='mask')
diff = 0
for i in range(len(inputs)):
    #diff += int(counts[i]) - np.sum(np.array(locations[i]))
    print(
        'img {}, counts: {}, sum: {}'.format(i, [ele[i] for ele in counts], np.sum(np.array(locations[i]))))

print('total difference :', diff)
