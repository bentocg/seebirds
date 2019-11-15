# modified from torchvision/datasets/folder.py

import torch.utils.data as data

from PIL import Image

import os
import os.path
import pandas as pd
import numpy as np


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def make_dataset(dir, extensions, training_set, shuffle=True, labels=['AdultBird', 'Chick', 'Egg']):
    images = []
    locations = {label: [] for label in labels}
    det_df = pd.read_csv('./training_sets/{}/centroids.csv'.format(training_set), index_col=0)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions):
                path = os.path.join(root, fname)
                images.append(path)
                # get locations
                
                if fname not in list(det_df.Image):
                    for label in labels:
                        locations[label].append ([])
                else:
                    for label in labels:
                        loc_str = det_df.loc[det_df.Image == fname][label].values[0]
                        if type(loc_str) == str:
                            locs = [int(ele) for ele in loc_str.split("_")]
                            locs = np.array([(locs[i+1], locs[i]) for i in range(0, len(locs)-1, 2)]).reshape(-1, 2)
                        else:
                            locs = []
                        locations[label].append(locs)
    # random permutation
    if shuffle:
        idcs = np.random.permutation([idx for idx in range(len(images))])
        images = [images[idx] for idx in idcs]
        for label in labels:
            locations[label] = [locations[label][idx] for idx in idcs]
    return [images, locations]


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader, extensions, training_set, shuffle=True, shape_transform=None, int_transform=None,
                 img_dim=[1080, 1920]):
        samples, locations = make_dataset(root, extensions, training_set, shuffle)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.img_dim = img_dim

        self.samples = samples
        self.locations = locations
        self.n_classes = len(locations)

        self.shape_transform = shape_transform
        self.int_transform = int_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path = self.samples[index]
        locs = [self.locations[label][index] for label in self.locations]
        sample = self.loader(path)
        hit_value = 255
       

        locations = np.zeros([self.img_dim[0], self.img_dim[1], self.n_classes], dtype=np.uint8)
        for idx, loc in enumerate(locs):
            for pnt in loc:
                if pnt[0] < self.img_dim[0] and pnt[1] < self.img_dim[1]:
                    locations[pnt[0], pnt[1], idx] = hit_value
        locations = Image.fromarray(locations.astype('uint8'))

        if self.shape_transform is not None:
            sample, locations, count = self.shape_transform(sample, locations)

        if self.int_transform is not None:
            sample = self.int_transform(sample)

        return sample, count, locations

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Shape Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.shape_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Intensity Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.int_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolderTrainDet(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, training_set, loader=default_loader, extensions=IMG_EXTENSIONS, shuffle=True, shape_transform=None, int_transform=None):
        super(ImageFolderTrainDet, self).__init__(root=root, training_set=training_set, loader=loader, extensions=extensions,
                                                  shuffle=shuffle, shape_transform=shape_transform,
                                                  int_transform=int_transform)
                    

