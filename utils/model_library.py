"""
Model library
==========================================================
Script to keep track of model architectures / hyperparameter sets used in experiments for ICEBERG seals use case.
Author: Bento Goncalves
License: MIT
Copyright: 2018-2019
"""

__all__ = ['model_archs', 'training_sets', 'hyperparameters', 'model_defs', 'dataloaders']


from utils.custom_architectures import *
from utils.dataloaders import *

# architecture definitions with input size and whether the model is used at the haulout level or single seal level
model_archs = {'Unet': {'input_size': 512, 'pipeline': 'Heatmap'},
               'UnetCntWRN': {'input_size': 224, 'pipeline': 'Heatmap-Cnt'},
               'UnetOccDense': {'input_size': 224, 'pipeline': 'Heatmap-Occ'},
               'UnetCntWRNOccDense': {'input_size': 224, 'pipeline': 'Heatmap-Cnt-Occ'},
               'SealNet_V10': {'input_size': 224, 'pipeline': 'Heatmap-Cnt-Occ'},
               }

# model definitions
model_defs = {'Heatmap': {'Unet': UNet(scale=32, n_channels=3, n_classes=3)},
              'Heatmap-Cnt': {'UnetCntWRN': UNetCntWRN(scale=32, depth=28)},
              'Heatmap-Occ': {'UnetOccDense': UNetOccDense(scale=32)},
              'Heatmap-Cnt-Occ': {'UnetCntWRNOccDense': UNetCntWRNOccDense(scale=32, depth=28),
                                  'SealNet_V10': SealNet_V10(scale=32, depth=28)}
}


# model dataloaders
dataloaders = {'Heatmap': lambda dataset, shp_trans, int_trans: ImageFolderTrainDet(dataset, shp_trans, int_trans),
               'Heatmap-Cnt': lambda dataset, shp_trans, int_trans: ImageFolderTrainDet(dataset, shp_trans, int_trans),
               'Heatmap-Occ': lambda dataset, shp_trans, int_trans: ImageFolderTrainDet(dataset, shp_trans, int_trans),
               'Heatmap-Cnt-Occ': lambda dataset, shp_trans, int_trans: ImageFolderTrainDet(dataset, shp_trans,
                                                                                            int_trans)
               }


# training sets with number of classes and size of scale bands
training_sets = {'training_set_vanilla': {'num_classes': 11, 'scale_bands': [512, 512, 512]},
                 'training_set_vanillaMS': {'num_classes': 11, 'scale_bands': [450, 1350, 4000]},
                 'training_set_binary': {'num_classes': 2, 'scale_bands': [450, 450, 450]},
                 'training_set_binaryMS': {'num_classes': 2, 'scale_bands': [450, 1350, 4000]},
                 'training_set_penguins': {'num_classes': 2, 'scale_bands': [450, 450, 450]},
                 'training_set_dangers': {'num_classes': 2, 'scale_bands': [450, 450, 450]},
                 'training_set_vanillaNS': {'num_classes:': 10, 'scale_bands': [450, 450, 450]}
                 }

# hyperparameter sets
hyperparameters = {'A': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 32, 'batch_size_test': 128,
                         'step_size': 1, 'gamma': 0.5, 'epochs': 150, 'num_workers_train': 16, 'num_workers_val': 16},
                   'B': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 128, 'batch_size_test': 8,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 5, 'num_workers_train':16, 'num_workers_val': 16},
                   'C': {'learning_rate': 1E-3, 'batch_size_train': 16, 'batch_size_val': 32, 'batch_size_test': 32,
                         'step_size': 1, 'gamma': 0.95, 'epochs': 75, 'num_workers_train': 8, 'num_workers_val': 8},
                   'D': {'learning_rate': 5E-4, 'batch_size_train': 64, 'batch_size_val': 128, 'batch_size_test': 128,
                         'step_size': 1, 'gamma': 0.5, 'epochs': 75, 'num_workers_train': 16, 'num_workers_val': 32},
                   'E': {'learning_rate': 1E-3, 'batch_size_train': 128, 'batch_size_val': 256, 'batch_size_test': 256,
                         'step_size': 1, 'gamma': 0.5, 'epochs': 75, 'num_workers_train': 32, 'num_workers_val': 64},
                   'F': {'learning_rate': 1E-3, 'batch_size_train': 64, 'batch_size_val': 128, 'batch_size_test': 128,
                         'step_size': 1, 'gamma': 0.5, 'epochs': 75, 'num_workers_train': 16, 'num_workers_val': 16}
                   }
