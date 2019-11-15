"""
Train sealnet
==========================================================

CNN training script for ICEBERG seals use case.

Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

import argparse
import datetime
import os
import shutil
import time
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms

from utils.dataloaders.data_loader_train_det import ImageFolderTrainDet
from utils.dataloaders.transforms_det import ShapeTransform
from utils.model_library import *
from utils.getxy_max import getxy_max

parser = argparse.ArgumentParser(description='trains a CNN to find seals in satellite imagery')
parser.add_argument('--training_dir', type=str, help='base directory to recursively search for images in')
parser.add_argument('--model_architecture', type=str, help='model architecture, must be a member of models '
                                                           'dictionary')
parser.add_argument('--hyperparameter_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                           'hyperparameters dictionary')
parser.add_argument('--output_name', type=str, help='name of output file from training, this name will also be used in '
                                                    'subsequent steps of the pipeline')
parser.add_argument('--models_folder', type=str, default='saved_models', help='folder where the model will be saved')
parser.add_argument('--all_train', type=int, default=0, help='whether all samples will be used for training')
parser.add_argument('--gpu_id', type=int, help='GPU id for running training script, defaults to nn.Dataparallel if id is not provided')

args = parser.parse_args()

# define pipeline
pipeline = model_archs[args.model_architecture]['pipeline']

# check for invalid inputs
if args.model_architecture not in model_archs:
    raise Exception("Invalid architecture -- see supported architectures:  {}".format(list(model_archs.keys())))

if args.training_dir not in training_sets:
    raise Exception("Training set is not defined in ./utils/model_library.py")

if args.hyperparameter_set not in hyperparameters:
    raise Exception("Hyperparameter combination is not defined in ./utils/model_library.py")

# image transforms seem to cause truncated images, so we need this
ImageFile.LOAD_TRUNCATED_IMAGES = True

# we get an RGB warning, but the loader properly converts to RGB -after- this
warnings.filterwarnings('ignore', module='PIL')

# Data augmentation and normalization for training
# Just normalization for validation
arch_input_size = model_archs[args.model_architecture]['input_size']

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

# define data dir and image size
data_dir = "./training_sets/{}".format(args.training_dir)
img_size = training_sets[args.training_dir]['scale_bands'][0]

# save image datasets
if args.all_train:
    image_datasets = {'training': ImageFolderTrainDet(root=data_dir,
                                             shape_transform=data_transforms['training']['shape_transform'],
                                             int_transform=data_transforms['training']['int_transform'],
                                             training_set=args.training_dir,
                                             shuffle=True)}
 

else:
    image_datasets = {x: ImageFolderTrainDet(root=os.path.join(data_dir, x),
                                             shape_transform=data_transforms[x]['shape_transform'],
                                             int_transform=data_transforms[x]['int_transform'],
                                             training_set=args.training_dir,
                                             shuffle=x == 'training')
                      for x in ['training', 'validation']}


# change batch size ot match number of GPU's being used?
dataloaders = {"training": torch.utils.data.DataLoader(image_datasets["training"],
                                                       batch_size=
                                                       hyperparameters[args.hyperparameter_set]['batch_size_train'],
                                                       num_workers=
                                                       hyperparameters[args.hyperparameter_set]['num_workers_train'],
                                                       shuffle=True)}
                                                    
if not args.all_train:
    dataloaders["validation"] =  torch.utils.data.DataLoader(image_datasets["validation"],
                                                         batch_size=
                                                         hyperparameters[args.hyperparameter_set]['batch_size_val'],
                                                         num_workers=
                                                         hyperparameters[args.hyperparameter_set]['num_workers_val'])
dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}

use_gpu = torch.cuda.is_available()

sigmoid = torch.nn.Sigmoid()


def save_checkpoint(state, is_best_loss, is_best_f1, is_best_recall, is_best_precision):
    """
    Saves model checkpoints during training. At the end of each validation cycle, model checkpoints are stored if they
    surpass previous best scores at a number of validation metrics (i.e. loss, F-1 score, precision and recall).
    Alongside the model state from the latest epoch, one model state is kept for each validation metric.

    :param state: pytorch model state, with parameter values for a giv
    :param is_best_loss: boolean for whether the current state beats the lowest loss
    :param is_best_f1: boolean for whether the current state beats the highest F-1 score
    :param is_best_recall: boolean for whether the current state beats the highest recall
    :param is_best_precision: boolean for whether the current state beats the highest precision
    :return:
    """
    filename = './{}/{}/{}/{}'.format(args.models_folder, pipeline, args.output_name, args.output_name)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    torch.save(state, filename + '.tar')
    if is_best_loss:
        shutil.copyfile(filename + '.tar', filename + '_best_loss.tar')
    if is_best_f1:
        shutil.copyfile(filename + '.tar', filename + '_best_f1.tar')
    if is_best_recall:
        shutil.copyfile(filename + '.tar', filename + '_best_recall.tar')
    if is_best_precision:
        shutil.copyfile(filename + '.tar', filename + '_best_precision.tar')


def train_model(model, criterion1, criterion2, criterion3, optimizer, scheduler, all_train, num_epochs=25, device=0):
    """
    Helper function to train CNNs. Trains detection models using heatmaps, where the output heatmap has the same
    dimensions of the input image. Heatmap detection may be assisted with a regression branch to provide counts and/or
    an occupancy branch to decide if a patch is worth counting.

    :param model: pytorch model
    :param criterion1: loss function
    :param criterion2: additional loss function
    :param criterion3: additional loss function
    :param optimizer: optimizer
    :param scheduler: training scheduler to deal with weight decay during training
    :param num_epochs: number of training epochs
    :return:
    """
    # keep track of running time
    since = time.time()

    # create summary writer for tensorboardX
    writer = SummaryWriter(log_dir='./tensorboard_logs/{}_{}'.format(args.output_name, str(datetime.datetime.now())))

    # keep track of training iterations
    global_step = 0

    # keep track of best accuracy
    best_loss = 1000000000
    best_f1 = 0
    best_recall = 0
    best_precision = 0

    # loss dictionary
    loss_dict = {'count': lambda x: criterion1(x, counts),
                 'heatmap': lambda x: criterion2(x.view(-1), locations.view(-1)) * 10}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            print('\n{} \n'.format(phase))
            if phase == 'training':
                model.train(True)  # Set model to training mode

                running_loss = {'count': 0,
                                'heatmap': 0}

                # Iterate over data.
                for iter, data in enumerate(dataloaders[phase]):

                    # get the inputs
                    inputs, counts, locations = data


                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda(device=f"cuda:{device}"))
                        counts = Variable(counts.cuda(device=f"cuda:{device}"))
                        locations = Variable(locations.cuda(device=f"cuda:{device}"))
                    else:
                        inputs, counts, locations = Variable(inputs), Variable(counts), Variable(locations)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward pass to get prediction dictionary
                    out_dict = model(inputs)

                    # get losses
                    batch_loss = {}
                    for key in out_dict:
                        batch_loss[key] = loss_dict[key](out_dict[key])
                        running_loss[key] += batch_loss[key].item() * len(counts)

                    # add losses up
                    loss = 0
                    for ele in batch_loss:
                        if out_dict[ele].requires_grad:
                            loss += batch_loss[ele]

                    loss.backward()

                    # clip gradients
                    #nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                    # step with optimizer
                    optimizer.step()
                    scheduler.step()
                    global_step += 1
                    if global_step % 3 == 0:
                         writer.add_scalar('learning_rate', scheduler.get_lr(), global_step=global_step)
                    
                        


            else:
                if all_train:
                    continue
                model.train(False)  # Set model to evaluate mode
                with torch.no_grad():
                    running_loss = {'count': 0,
                                    'heatmap': 0}
                    false_negatives = 0
                    false_positives = 0
                    true_positives = 0
                    false_negatives_fixed = 0
                    false_positives_fixed = 0
                    true_positives_fixed = 0

                    # Iterate over data.
                    for iter, data in enumerate(dataloaders[phase]):
                        # get the inputs
                        inputs, counts, locations = data

                        # get precision and recall
                        #ground_truth_xy = [getxy_max(loc, int(counts[idx])) for idx, loc in
                        #                   enumerate(locations.numpy())]

                    
                        # wrap them in Variable
                        if use_gpu:
                            inputs = Variable(inputs.cuda(device=f"cuda:{device}"))
                            counts = Variable(counts.cuda(device=f"cuda:{device}"))
                            locations = Variable(locations.cuda(device=f"cuda:{device}"))
                        else:
                            inputs, counts, locations = Variable(inputs), Variable(counts), Variable(locations)

                        # forward
                        out_dict = model(inputs)

                        # get predicted locations, precision and recall
                        #pred_xy = [getxy_max(loc, int(round(
                        #    out_dict['count'][idx].item()))) for idx, loc in
                        #           enumerate(out_dict['heatmap'].cpu().numpy())]

                        #for idx, ele in enumerate(ground_truth_xy):
                        #    n_matches = 0
                        #    n_matches_fixed = 0
                        #    if len(ele) == 0:
                        #        false_positives += len(pred_xy[idx])
                        #    else:
                        #        matched_gt = set([])
                        #        matched_pred = set([])

                        #        for gt_idx, pnt in enumerate(ele):
                        #            pnt = np.array(pnt)
                        #
                        #            for pred_idx, pnt2 in enumerate(pred_xy[idx]):
                        #                pnt2 = np.array(pnt2)
                        #                if gt_idx in matched_gt:
                        #                    continue
                        #                if pred_idx not in matched_pred and np.linalg.norm(pnt - pnt2) < 3:
                        #                    n_matches += 1
                        #                    matched_pred.add(pred_idx)
                        #                    matched_gt.add(gt_idx)

                        #        true_positives += n_matches
                        #        false_positives += len(pred_xy[idx]) - n_matches
                        #        false_negatives += len(ele) - n_matches

                        # get losses
                        batch_loss = {}
                        for key in out_dict:
                            batch_loss[key] = loss_dict[key](out_dict[key])
                            running_loss[key] += batch_loss[key].item() * len(counts)

            # get epoch losses
            epoch_loss = {'heatmap': 0,
                          'count': 0}

            for loss in running_loss:
                epoch_loss[loss] = running_loss[loss] / dataset_sizes[phase]

            if phase == 'validation':
                #epoch_precision = true_positives / max(1, true_positives + false_positives)
                #epoch_recall = true_positives / max(1, true_positives + false_negatives)
                #epoch_f1 = epoch_precision * epoch_recall
          
                for loss in epoch_loss:
                    writer.add_scalar('validation_loss_{}'.format(loss), epoch_loss[loss], global_step=global_step)
                #writer.add_scalar('validation_precision', epoch_precision, global_step=global_step)
                #writer.add_scalar('validation_recall', epoch_recall, global_step=global_step)
                #writer.add_scalar('validation_f1', epoch_f1, global_step=global_step)
                total_loss = sum(epoch_loss.values())
                is_best_loss = total_loss < best_loss
                #is_best_f1 = epoch_f1 > best_f1
                #is_best_precision = epoch_precision > best_precision
                #is_best_recall = epoch_recall > best_recall
                best_loss = min(total_loss, best_loss)
                #best_f1 = max(epoch_f1, best_f1)
                save_checkpoint(model.state_dict(), is_best_loss, False, False, False)

            else:
                if all_train:
                    save_checkpoint(model.state_dict(), *[False] * 4)
                for loss in epoch_loss:
                    writer.add_scalar('training_loss_{}'.format(loss), epoch_loss[loss], global_step=global_step)
                

            for loss in epoch_loss:
                print('{} loss: {}'.format(loss, epoch_loss[loss]))

            if phase == 'validation':
                time_elapsed = time.time() - since
                print('training time: {}h {:.0f}m {:.0f}s\n'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60,
                                                                    time_elapsed % 60))
    time_elapsed = time.time() - since
    print('Training complete in {}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))


def main():
    # set seed
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = model_defs[pipeline][args.model_architecture]

    # define criterion
    criterion = nn.SmoothL1Loss()
    criterion2 = nn.BCEWithLogitsLoss()
    criterion3 = nn.BCEWithLogitsLoss()

    # find BCE weight
    #bce_weights = [arch_input_size ** 2 * (86514 / 232502), 11 / 2]

    if use_gpu:
        if args.gpu_id is None:
            model = nn.DataParallel(model.cuda())
            criterion = criterion.cuda()
            criterion2 = criterion2.cuda()
            criterion3 = criterion3.cuda()
        else:
            model.to(f"cuda:{args.gpu_id}")
            criterion.to(f"cuda:{args.gpu_id}")
            criterion2.to(f"cuda:{args.gpu_id}")
            criterion3.to(f"cuda:{args.gpu_id}")

    # Observe that all parameters are being optimized
    #optimizer_ft = optim.Adam(model.parameters(), lr=hyperparameters[args.hyperparameter_set]['learning_rate'])
    optimizer_ft = optim.AdamW(model.parameters(), lr=hyperparameters[args.hyperparameter_set]['learning_rate'])

    # Cosine Annealing scheduler
    #scheduler = lr_scheduler.OneCycleLR(optimizer_ft, max_lr=0.001, steps_per_epoch=len(dataloaders['training']), epochs=hyperparameters[args.hyperparameter_set]['epochs'])
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer_ft, eta_min=1E-5, T_max=len(dataloaders['training']) * (hyperparameters[args.hyperparameter_set]['epochs']))

    # start training
    train_model(model, criterion, criterion2, criterion3, optimizer_ft, scheduler, all_train=args.all_train,
                num_epochs=hyperparameters[args.hyperparameter_set]['epochs'])


if __name__ == '__main__':
    main()
